use chrono::{DateTime, Local};
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;
use windows::core::HSTRING;
use windows::Win32::Media::MediaFoundation::{
    IMFMediaBuffer, IMFMediaType, IMFSample, IMFSinkWriter, MFAudioFormat_AAC, MFAudioFormat_PCM,
    MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample, MFCreateSinkWriterFromURL,
    MFMediaType_Audio, MFMediaType_Video, MFStartup, MFVideoFormat_H264, MFVideoFormat_RGB32,
    MFVideoInterlace_Progressive, MFVideoPrimaries_BT709, MFVideoTransFunc_709, MFSTARTUP_FULL,
    MF_MT_AUDIO_AVG_BYTES_PER_SECOND, MF_MT_AUDIO_BITS_PER_SAMPLE, MF_MT_AUDIO_BLOCK_ALIGNMENT,
    MF_MT_AUDIO_NUM_CHANNELS, MF_MT_AUDIO_SAMPLES_PER_SECOND, MF_MT_AVG_BITRATE, MF_MT_FRAME_RATE,
    MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_PIXEL_ASPECT_RATIO,
    MF_MT_SUBTYPE, MF_MT_TRANSFER_FUNCTION, MF_MT_VIDEO_PRIMARIES, MF_SDK_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_APARTMENTTHREADED, COINIT_MULTITHREADED,
};

#[cfg(feature = "audio_capture")]
use windows::Win32::Media::Audio::{
    eCapture, eConsole, eRender, IAudioCaptureClient, IAudioClient, IMMDeviceEnumerator,
    MMDeviceEnumerator, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK,
};

// --- START: VOLUME CONTROL CONSTANTS ---
// Adjust these values to change the volume balance in the final video.
// 2.0 = 200% volume (louder), 0.5 = 50% volume (quieter)
const MIC_VOLUME: f32 = 2.5;
const SYSTEM_VOLUME: f32 = 0.15;
// --- END: VOLUME CONTROL CONSTANTS ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSource {
    Microphone,
    System,
    Both,
}

#[derive(Debug, Clone)]
pub struct Mp4SegmentConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output_dir: PathBuf,
    pub base_name: String,
    pub bitrate_kbps: u32,
    pub include_audio: bool,
    pub audio_bitrate_kbps: u32,
    pub audio_source: AudioSource,
}

pub struct Mp4SegmentWriter {
    sink: IMFSinkWriter,
    stream_index: u32,
    audio_stream_index: Option<u32>,
    frame_index: u64,
    width: u32,
    height: u32,
    fps: u32,
    audio_sample_rate: u32,
    audio_channels: u32,
    audio_time_100ns: u64,
    start_instant: Instant,
    last_video_time_100ns: u64,
    #[cfg(feature = "audio_capture")]
    _system_audio_thread: Option<std::thread::JoinHandle<()>>,
    #[cfg(feature = "audio_capture")]
    _mic_audio_thread: Option<std::thread::JoinHandle<()>>,
    #[cfg(feature = "audio_capture")]
    audio_stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>,
    system_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
    mic_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
    audio_source: AudioSource,
    pub output_path: PathBuf,
}

static INIT: Once = Once::new();

fn init_media_foundation() -> anyhow::Result<()> {
    let mut err: Option<anyhow::Error> = None;
    INIT.call_once(|| unsafe {
        let hr = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
        if hr.is_err() {
            err = Some(anyhow::anyhow!("CoInitializeEx failed: {:?}", hr));
            return;
        }
        let hr = MFStartup(MF_SDK_VERSION, MFSTARTUP_FULL);
        if hr.is_err() {
            err = Some(anyhow::anyhow!("MFStartup failed: {:?}", hr));
            return;
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    Ok(())
}

impl Mp4SegmentWriter {
    pub fn create_new(cfg: Mp4SegmentConfig) -> anyhow::Result<Self> {
        init_media_foundation()?;
        std::fs::create_dir_all(&cfg.output_dir)?;

        log::info!(
            "Creating MP4 writer - Audio enabled: {}, Source: {:?}",
            cfg.include_audio,
            cfg.audio_source
        );

        let now: DateTime<Local> = Local::now();
        let filename = format!("{}_{}.mp4", cfg.base_name, now.format("%Y%m%d_%H%M%S"));
        let path = cfg.output_dir.join(filename);
        let path_h = HSTRING::from(path.to_string_lossy().to_string());

        let (sink, stream_index, audio_stream_index_opt) = unsafe {
            let sink: IMFSinkWriter = MFCreateSinkWriterFromURL(&path_h, None, None)?;

            // Video Output Type (H.264)
            let out_type = MFCreateMediaType()?;
            out_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            out_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)?;
            out_type.SetUINT32(&MF_MT_AVG_BITRATE, cfg.bitrate_kbps * 1000)?;
            set_mt_size(&out_type, &MF_MT_FRAME_SIZE, cfg.width, cfg.height)?;
            set_mt_ratio(&out_type, &MF_MT_FRAME_RATE, cfg.fps, 1)?;
            out_type.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
            let stream_index = sink.AddStream(&out_type)?;

            // Video Input Type (RGB32/BGRA)
            let in_type = MFCreateMediaType()?;
            in_type.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            in_type.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
            set_mt_size(&in_type, &MF_MT_FRAME_SIZE, cfg.width, cfg.height)?;
            set_mt_ratio(&in_type, &MF_MT_FRAME_RATE, cfg.fps, 1)?;
            set_mt_ratio(&in_type, &MF_MT_PIXEL_ASPECT_RATIO, 1, 1)?;
            in_type.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
            in_type.SetUINT32(&MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709.0 as u32)?;
            in_type.SetUINT32(&MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_709.0 as u32)?;
            sink.SetInputMediaType(stream_index, &in_type, None)?;

            // Audio Stream Setup
            let audio_channels: u32 = 2;
            let audio_sample_rate: u32 = 48_000;
            let mut audio_stream_index_opt: Option<u32> = None;

            if cfg.include_audio {
                log::info!(
                    "Setting up audio stream: {}Hz, {} channels",
                    audio_sample_rate,
                    audio_channels
                );

                // Audio Output Type (AAC)
                let out_a = MFCreateMediaType()?;
                out_a.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)?;
                out_a.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC)?;
                out_a.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, audio_channels)?;
                out_a.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, audio_sample_rate)?;
                out_a.SetUINT32(&MF_MT_AVG_BITRATE, cfg.audio_bitrate_kbps * 1000)?;
                let audio_stream_index = sink.AddStream(&out_a)?;

                // Audio Input Type (PCM 16-bit)
                let in_a = MFCreateMediaType()?;
                in_a.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)?;
                in_a.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM)?;
                in_a.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, audio_channels)?;
                in_a.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, audio_sample_rate)?;
                in_a.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16)?;
                let block_align = audio_channels * 2;
                in_a.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align)?;
                in_a.SetUINT32(
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    audio_sample_rate * block_align,
                )?;
                sink.SetInputMediaType(audio_stream_index, &in_a, None)?;
                audio_stream_index_opt = Some(audio_stream_index);

                log::info!("Audio stream configured successfully");
            }

            sink.BeginWriting()?;
            (sink, stream_index, audio_stream_index_opt)
        };

        let mut writer = Self {
            sink,
            stream_index,
            audio_stream_index: audio_stream_index_opt,
            frame_index: 0,
            width: cfg.width,
            height: cfg.height,
            fps: cfg.fps,
            audio_sample_rate: 48_000,
            audio_channels: 2,
            audio_time_100ns: 0,
            start_instant: Instant::now(),
            last_video_time_100ns: 0,
            #[cfg(feature = "audio_capture")]
            _system_audio_thread: None,
            #[cfg(feature = "audio_capture")]
            _mic_audio_thread: None,
            #[cfg(feature = "audio_capture")]
            audio_stop_signal: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            system_buffer: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            mic_buffer: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            audio_source: cfg.audio_source,
            output_path: path,
        };

        // Start audio capture if enabled
        #[cfg(feature = "audio_capture")]
        if cfg.include_audio {
            log::info!("Attempting to start audio capture...");
            match writer.start_audio_capture(cfg.audio_source) {
                Ok(_) => log::info!("Audio capture initialization completed"),
                Err(e) => log::error!("Failed to start audio capture: {:?}", e),
            }
        }

        #[cfg(not(feature = "audio_capture"))]
        if cfg.include_audio {
            log::warn!("Audio requested but audio_capture feature not enabled!");
        }

        Ok(writer)
    }

    pub fn encode_rgb_frame(&mut self, rgb: &[u8]) -> anyhow::Result<()> {
        let elapsed_100ns: u64 = self.start_instant.elapsed().as_nanos().saturating_div(100) as u64;
        let nominal_frame_100ns: u64 = 10_000_000u64 / (self.fps.max(1) as u64);
        let target_video_time = if elapsed_100ns > self.last_video_time_100ns {
            elapsed_100ns
        } else {
            self.last_video_time_100ns
                .saturating_add(nominal_frame_100ns)
        };

        // Convert RGB -> BGRA
        let mut bgra_bytes: Vec<u8> = Vec::with_capacity(rgb.len() / 3 * 4);
        for px in rgb.chunks_exact(3) {
            bgra_bytes.push(px[2]); // B
            bgra_bytes.push(px[1]); // G
            bgra_bytes.push(px[0]); // R
            bgra_bytes.push(255); // A
        }

        unsafe {
            let sample: IMFSample = MFCreateSample()?;
            let buffer: IMFMediaBuffer = MFCreateMemoryBuffer(bgra_bytes.len() as u32)?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            buffer.Lock(&mut ptr, None, None)?;
            std::ptr::copy_nonoverlapping(bgra_bytes.as_ptr(), ptr, bgra_bytes.len());
            buffer.Unlock()?;
            buffer.SetCurrentLength(bgra_bytes.len() as u32)?;
            sample.AddBuffer(&buffer)?;

            let video_start_100ns = self.last_video_time_100ns;
            let video_dur_100ns = (target_video_time.saturating_sub(video_start_100ns)).max(1);
            sample.SetSampleTime(video_start_100ns as i64)?;
            sample.SetSampleDuration(video_dur_100ns as i64)?;

            self.sink.WriteSample(self.stream_index, &sample)?;
            self.frame_index += 1;
        }

        // Write audio to maintain sync
        self.write_mixed_audio_until(target_video_time)?;
        self.last_video_time_100ns = target_video_time;
        Ok(())
    }

    pub fn finalize(mut self) -> anyhow::Result<PathBuf> {
        log::info!("Finalizing MP4 file with {} frames", self.frame_index);

        // Stop audio threads
        #[cfg(feature = "audio_capture")]
        {
            self.audio_stop_signal
                .store(true, std::sync::atomic::Ordering::SeqCst);

            if let Some(thread) = self._system_audio_thread.take() {
                let _ = thread.join();
            }
            if let Some(thread) = self._mic_audio_thread.take() {
                let _ = thread.join();
            }
        }

        unsafe {
            let _ = self.sink.Finalize();
        }
        if self.frame_index == 0 {
            let _ = std::fs::remove_file(&self.output_path);
        }
        Ok(self.output_path)
    }

    #[cfg(feature = "audio_capture")]
    fn start_audio_capture(&mut self, source: AudioSource) -> anyhow::Result<()> {
        log::info!("Starting audio capture with source: {:?}", source);

        // Start WASAPI system audio capture (loopback)
        if source == AudioSource::System || source == AudioSource::Both {
            let buffer = self.system_buffer.clone();
            let stop_signal = self.audio_stop_signal.clone();

            let thread = std::thread::spawn(move || {
                if let Err(e) = capture_system_audio_wasapi(stop_signal, buffer) {
                    log::error!("System audio capture failed: {:?}", e);
                }
            });

            self._system_audio_thread = Some(thread);
            log::info!("✓ Started WASAPI system audio loopback thread");
        }

        // Start WASAPI microphone capture
        if source == AudioSource::Microphone || source == AudioSource::Both {
            let buffer = self.mic_buffer.clone();
            let stop_signal = self.audio_stop_signal.clone();

            let thread = std::thread::spawn(move || {
                if let Err(e) = capture_microphone_wasapi(stop_signal, buffer) {
                    log::error!("Microphone capture failed: {:?}", e);
                }
            });

            self._mic_audio_thread = Some(thread);
            log::info!("✓ Started WASAPI microphone capture thread");
        }

        Ok(())
    }

    fn write_mixed_audio_until(&mut self, target_time_100ns: u64) -> anyhow::Result<()> {
        let Some(audio_stream) = self.audio_stream_index else {
            return Ok(());
        };
        if target_time_100ns <= self.audio_time_100ns {
            return Ok(());
        }

        let dur_100ns = target_time_100ns - self.audio_time_100ns;
        let samples_needed = (self.audio_sample_rate as u64 * dur_100ns) / 10_000_000u64;
        if samples_needed == 0 {
            return Ok(());
        }

        let channels = self.audio_channels as usize;
        let samples_needed_total = (samples_needed as usize) * channels;

        // Get audio from appropriate source(s) and mix
        let audio_data = match self.audio_source {
            AudioSource::System => {
                let mut sys = self.system_buffer.lock().unwrap();
                let to_take = sys.len().min(samples_needed_total);
                let mut data: Vec<i16> = sys.drain(..to_take).collect();
                data.resize(samples_needed_total, 0);
                data
            }
            AudioSource::Microphone => {
                let mut mic = self.mic_buffer.lock().unwrap();
                let to_take = mic.len().min(samples_needed_total);
                let mut data: Vec<i16> = mic.drain(..to_take).collect();
                data.resize(samples_needed_total, 0);
                data
            }
            AudioSource::Both => {
                // PROPER MIXING: Weighted mix with volume control
                let mut sys = self.system_buffer.lock().unwrap();
                let mut mic = self.mic_buffer.lock().unwrap();

                let sys_take = sys.len().min(samples_needed_total);
                let mic_take = mic.len().min(samples_needed_total);

                let sys_data: Vec<i16> = sys.drain(..sys_take).collect();
                let mic_data: Vec<i16> = mic.drain(..mic_take).collect();

                let max_len = sys_data.len().max(mic_data.len()).max(samples_needed_total);
                let mut mixed = Vec::with_capacity(max_len);

                for i in 0..max_len {
                    let sys_sample = sys_data.get(i).copied().unwrap_or(0) as f32;
                    let mic_sample = mic_data.get(i).copied().unwrap_or(0) as f32;
                    
                    // Apply volume multipliers and mix
                    let mixed_f32 = (sys_sample * SYSTEM_VOLUME) + (mic_sample * MIC_VOLUME);
                    let mixed_sample = mixed_f32.clamp(-32768.0, 32767.0) as i16;
                    mixed.push(mixed_sample);
                }

                mixed
            }
        };

        // Convert to bytes and write
        let mut bytes = Vec::with_capacity(audio_data.len() * 2);
        for &sample in &audio_data {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        unsafe {
            let sample: IMFSample = MFCreateSample()?;
            let buffer: IMFMediaBuffer = MFCreateMemoryBuffer(bytes.len() as u32)?;
            let mut ptr: *mut u8 = std::ptr::null_mut();
            buffer.Lock(&mut ptr, None, None)?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
            buffer.Unlock()?;
            buffer.SetCurrentLength(bytes.len() as u32)?;
            sample.AddBuffer(&buffer)?;
            sample.SetSampleTime(self.audio_time_100ns as i64)?;
            sample.SetSampleDuration(dur_100ns as i64)?;
            self.sink.WriteSample(audio_stream, &sample)?;
        }

        self.audio_time_100ns += dur_100ns;
        Ok(())
    }
}

// WASAPI System Audio Capture (Loopback)
#[cfg(feature = "audio_capture")]
fn capture_system_audio_wasapi(
    stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>,
    mix_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
) -> anyhow::Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

        // Get default output device
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let wave_format_ptr = audio_client.GetMixFormat()?;
        let wave_format = *wave_format_ptr;

        let sample_rate = wave_format.nSamplesPerSec;
        let channels = wave_format.nChannels;
        let bits_per_sample = wave_format.wBitsPerSample;

        log::info!(
            "System audio format: {}Hz, {} channels, {} bits",
            sample_rate,
            channels,
            bits_per_sample
        );

        audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_LOOPBACK,
            200_000_000, // 2 seconds buffer
            0,
            wave_format_ptr,
            None,
        )?;

        CoTaskMemFree(Some(wave_format_ptr as *const _));

        let capture_client: IAudioCaptureClient = audio_client.GetService()?;
        audio_client.Start()?;

        log::info!("✓ WASAPI system audio capture started");

        let max_samples = 288_000; // 3 seconds @ 48kHz stereo

        let mut prev_sample: i16 = 0;
        while !stop_signal.load(std::sync::atomic::Ordering::SeqCst) {
            let packet_size = capture_client.GetNextPacketSize()?;
            if packet_size == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            let mut data_ptr = std::ptr::null_mut();
            let mut num_frames = 0;
            let mut flags = 0;
            capture_client.GetBuffer(&mut data_ptr, &mut num_frames, &mut flags, None, None)?;

            if num_frames > 0 {
                let ch = channels as usize;
                let num_samples = num_frames as usize * ch;

                let mut buf = mix_buffer.lock().unwrap();

                if bits_per_sample == 32 {
                    // Float32 PCM — most common for system loopback
                    let samples = std::slice::from_raw_parts(data_ptr as *const f32, num_samples);
                    for &sample in samples {
                        let s = (sample * 1.2).clamp(-1.0, 1.0);
                        let mut s16 = if s >= 0.0 {
                            (s * 32767.0) as i16
                        } else {
                            (s * 32768.0) as i16
                        };

                        // Simple low-pass filter to smooth hiss
                        s16 = ((prev_sample as f32 * 0.3) + (s16 as f32 * 0.7)) as i16;
                        prev_sample = s16;

                        buf.push(s16);
                    }
                } else if bits_per_sample == 16 {
                    let samples = std::slice::from_raw_parts(data_ptr as *const i16, num_samples);
                    buf.extend_from_slice(samples);
                } else {
                    log::warn!("Unsupported PCM bit depth: {}", bits_per_sample);
                }

                if buf.len() > max_samples {
                    let len = buf.len();
                    let excess = len - max_samples;
                    buf.drain(0..excess);
                }

                capture_client.ReleaseBuffer(num_frames)?;
            }
        }

        audio_client.Stop()?;
        CoUninitialize();
        log::info!("WASAPI system audio capture stopped");
    }

    Ok(())
}

// WASAPI Microphone Capture
#[cfg(feature = "audio_capture")]
fn capture_microphone_wasapi(
    stop_signal: std::sync::Arc<std::sync::atomic::AtomicBool>,
    mix_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
) -> anyhow::Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
        let device = enumerator.GetDefaultAudioEndpoint(eCapture, eConsole)?;

        let audio_client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let wave_format_ptr = audio_client.GetMixFormat()?;
        let wave_format = *wave_format_ptr;

        let sample_rate = wave_format.nSamplesPerSec;
        let channels = wave_format.nChannels;
        let bits_per_sample = wave_format.wBitsPerSample;

        log::info!(
            "Microphone format: {}Hz, {} channels, {} bits",
            sample_rate,
            channels,
            bits_per_sample
        );

        audio_client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            0, // No special flags for microphone
            200_000_000,
            0,
            wave_format_ptr,
            None,
        )?;

        CoTaskMemFree(Some(wave_format_ptr as *const _));

        let capture_client: IAudioCaptureClient = audio_client.GetService()?;
        audio_client.Start()?;

        log::info!("✓ WASAPI microphone capture started");

        let max_samples = 288_000;

        while !stop_signal.load(std::sync::atomic::Ordering::SeqCst) {
            let packet_size = capture_client.GetNextPacketSize()?;
            if packet_size == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }

            let mut data_ptr = std::ptr::null_mut();
            let mut num_frames = 0;
            let mut flags = 0;

            capture_client.GetBuffer(&mut data_ptr, &mut num_frames, &mut flags, None, None)?;

            if num_frames > 0 {
                let ch = channels as usize;
                let num_samples = num_frames as usize * ch;
                let samples_slice = std::slice::from_raw_parts(data_ptr as *const f32, num_samples);

                let mut buf = mix_buffer.lock().unwrap();
                let mut prev = 0i16;

                // Convert to stereo if mono
                if ch == 1 {
                    for &sample in samples_slice {
                        let s = (sample * 1.2).clamp(-1.0, 1.0);
                        let s16 = if s >= 0.0 {
                            (s * 32767.0) as i16
                        } else {
                            (s * 32768.0) as i16
                        };

                        // Simple low-pass smoothing filter
                        let filtered = ((prev as f32 * 0.3) + (s16 as f32 * 0.7)) as i16;
                        prev = filtered;

                        // Duplicate for stereo
                        buf.push(filtered);
                        buf.push(filtered);
                    }
                } else {
                    for &sample in samples_slice {
                        let s = (sample * 1.2).clamp(-1.0, 1.0);
                        let s16 = if s >= 0.0 {
                            (s * 32767.0) as i16
                        } else {
                            (s * 32768.0) as i16
                        };
                        buf.push(s16);
                    }
                }

                if buf.len() > max_samples {
                    let excess = buf.len() - max_samples;
                    buf.drain(0..excess);
                }

                capture_client.ReleaseBuffer(num_frames)?;
            }
        }

        audio_client.Stop()?;
        CoUninitialize();
        log::info!("WASAPI microphone capture stopped");
    }

    Ok(())
}

// Backward compatibility
impl Mp4SegmentWriter {
    #[cfg(feature = "audio_capture")]
    pub fn start_mic_capture(&mut self) -> anyhow::Result<()> {
        self.start_audio_capture(AudioSource::Microphone)
    }
}

use windows::core::Interface;
use windows::Win32::Media::MediaFoundation::IMFAttributes;

fn set_mt_size(
    mt: &IMFMediaType,
    key: &windows::core::GUID,
    w: u32,
    h: u32,
) -> windows::core::Result<()> {
    let attrs: IMFAttributes = mt.cast()?;
    let value: u64 = ((w as u64) << 32) | (h as u64);
    unsafe { attrs.SetUINT64(key, value) }
}

fn set_mt_ratio(
    mt: &IMFMediaType,
    key: &windows::core::GUID,
    num: u32,
    den: u32,
) -> windows::core::Result<()> {
    let attrs: IMFAttributes = mt.cast()?;
    let value: u64 = ((num as u64) << 32) | (den as u64);
    unsafe { attrs.SetUINT64(key, value) }
}