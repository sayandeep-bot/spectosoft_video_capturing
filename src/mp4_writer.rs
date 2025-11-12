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
use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};

#[cfg(feature = "audio_capture")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

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
    _audio_streams: Vec<cpal::Stream>,
    mix_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
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

                // Audio Input Type (PCM)
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
            _audio_streams: Vec::new(),
            mix_buffer: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
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

    pub fn finalize(self) -> anyhow::Result<PathBuf> {
        log::info!("Finalizing MP4 file with {} frames", self.frame_index);
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
        use cpal::{InputCallbackInfo, SampleFormat, StreamConfig, StreamError};

        log::info!("Starting audio capture with source: {:?}", source);

        let host = cpal::default_host();
        let max_buffer_samples = (3 * self.audio_sample_rate as usize) * 2;

        let build_stream = |device: &cpal::Device,
                            mix_buffer: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
                            device_name: String|
         -> anyhow::Result<cpal::Stream> {
            let cfg_any = device.default_input_config()?;
            let channels = cfg_any.channels() as usize;
            let scfg: StreamConfig = cfg_any.clone().into();

            log::info!(
                "Starting {} audio: {} channels, {:?}, {}Hz",
                device_name,
                channels,
                cfg_any.sample_format(),
                cfg_any.sample_rate().0
            );

            let device_name_clone = device_name.clone();
            let err_cb = move |err: StreamError| {
                log::error!("{} audio stream error: {:?}", device_name_clone, err);
            };

            let stream = match cfg_any.sample_format() {
                SampleFormat::I16 => {
                    let mix = mix_buffer.clone();
                    let data_cb = move |data: &[i16], _info: &InputCallbackInfo| {
                        let mut buf = mix.lock().unwrap();
                        if channels == 2 {
                            buf.extend_from_slice(data);
                        } else {
                            for &s in data {
                                buf.push(s);
                                buf.push(s);
                            }
                        }
                        if buf.len() > max_buffer_samples {
                            let excess = buf.len().saturating_sub(max_buffer_samples);
                            if excess > 0 {
                                buf.drain(0..excess);
                            }
                        }
                    };
                    device.build_input_stream::<i16, _, _>(&scfg, data_cb, err_cb, None)?
                }
                SampleFormat::U16 => {
                    let mix = mix_buffer.clone();
                    let data_cb = move |data: &[u16], _info: &InputCallbackInfo| {
                        let mut buf = mix.lock().unwrap();
                        if channels == 2 {
                            for &s in data {
                                buf.push((s as i32 - 32768) as i16);
                            }
                        } else {
                            for &s in data {
                                let v = (s as i32 - 32768) as i16;
                                buf.push(v);
                                buf.push(v);
                            }
                        }
                        if buf.len() > max_buffer_samples {
                            let excess = buf.len().saturating_sub(max_buffer_samples);
                            if excess > 0 {
                                buf.drain(0..excess);
                            }
                        }
                    };
                    device.build_input_stream::<u16, _, _>(&scfg, data_cb, err_cb, None)?
                }
                SampleFormat::F32 => {
                    let mix = mix_buffer.clone();
                    let data_cb = move |data: &[f32], _info: &InputCallbackInfo| {
                        let mut buf = mix.lock().unwrap();
                        if channels == 2 {
                            for &s in data {
                                buf.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
                            }
                        } else {
                            for &s in data {
                                let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                                buf.push(v);
                                buf.push(v);
                            }
                        }
                        if buf.len() > max_buffer_samples {
                            let excess = buf.len().saturating_sub(max_buffer_samples);
                            if excess > 0 {
                                buf.drain(0..excess);
                            }
                        }
                    };
                    device.build_input_stream::<f32, _, _>(&scfg, data_cb, err_cb, None)?
                }
                _ => return Err(anyhow::anyhow!("Unsupported sample format")),
            };

            stream.play()?;
            log::info!("{} audio stream started successfully", device_name);
            Ok(stream)
        };

        let mut streams = Vec::new();

        // Microphone
        if source == AudioSource::Microphone || source == AudioSource::Both {
            if let Some(mic_device) = host.default_input_device() {
                let name = mic_device
                    .name()
                    .unwrap_or_else(|_| "Unknown Mic".to_string());
                log::info!("Found microphone device: {}", name);
                match build_stream(
                    &mic_device,
                    self.mix_buffer.clone(),
                    format!("Microphone ({})", name),
                ) {
                    Ok(stream) => streams.push(stream),
                    Err(e) => log::warn!("Failed to start microphone: {:?}", e),
                }
            } else {
                log::warn!("No default microphone found");
            }
        }

        // System audio (loopback)
        if source == AudioSource::System || source == AudioSource::Both {
            log::info!("Searching for system audio (loopback) devices...");
            let mut found_loopback = false;

            if let Ok(devices) = host.input_devices() {
                let devices_vec: Vec<_> = devices.collect();
                log::info!("Found {} input devices total", devices_vec.len());

                for device in devices_vec {
                    if let Ok(name) = device.name() {
                        log::debug!("Checking device: {}", name);
                        let name_lower = name.to_lowercase();

                        if name_lower.contains("stereo mix")
                            || name_lower.contains("wave out mix")
                            || name_lower.contains("what u hear")
                            || name_lower.contains("loopback")
                            || name_lower.contains("what you hear")
                        {
                            log::info!("Found loopback device: {}", name);
                            match build_stream(
                                &device,
                                self.mix_buffer.clone(),
                                format!("System Audio ({})", name),
                            ) {
                                Ok(stream) => {
                                    streams.push(stream);
                                    found_loopback = true;
                                    break;
                                }
                                Err(e) => log::warn!(
                                    "Failed to start system audio from {}: {:?}",
                                    name,
                                    e
                                ),
                            }
                        }
                    }
                }
            }

            if !found_loopback {
                log::warn!("═══════════════════════════════════════════════════════════");
                log::warn!("System audio (loopback) NOT AVAILABLE");
                log::warn!("To enable system audio capture:");
                log::warn!("1. Right-click speaker icon → 'Sounds'");
                log::warn!("2. 'Recording' tab → Right-click → 'Show Disabled Devices'");
                log::warn!("3. Enable 'Stereo Mix' or 'Wave Out Mix'");
                log::warn!("4. Set it as default or at least enable it");
                log::warn!("OR install VB-Cable: https://vb-audio.com/Cable/");
                log::warn!("═══════════════════════════════════════════════════════════");
            }
        }

        self._audio_streams = streams;

        if self._audio_streams.is_empty() {
            log::error!("No audio streams started - audio will NOT be recorded!");
            return Err(anyhow::anyhow!("No audio devices available"));
        } else {
            log::info!(
                "✓ Successfully started {} audio stream(s)",
                self._audio_streams.len()
            );
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

        let mut mix = self.mix_buffer.lock().unwrap();
        let available = mix.len();

        let to_take = available.min(samples_needed_total);
        let mut audio_data: Vec<i16> = mix.drain(..to_take).collect();

        if audio_data.len() < samples_needed_total {
            audio_data.resize(samples_needed_total, 0);
        }

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
