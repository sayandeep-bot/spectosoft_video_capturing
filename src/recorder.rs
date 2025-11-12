use crate::avi_writer::{AviSegmentConfig, AviSegmentWriter};
use crate::mp4_writer::AudioSource;
use crate::mp4_writer::{Mp4SegmentConfig, Mp4SegmentWriter};
use crate::screenshotter::{Screenshotter, ScreenshotterConfig};
#[cfg(feature = "webm")]
use crate::webm_writer::{WebmSegmentConfig, WebmSegmentWriter};
use chrono::{DateTime, Local};
use image::{codecs::jpeg::JpegEncoder, ColorType};
use log::{error, warn};
use screenshots::Screen;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct RecorderConfig {
    pub output_dir: PathBuf,
    pub base_name: String,
    pub segment_duration: Duration,
    pub fps: u32,
    pub screenshot_interval: Duration,
    pub container: Container,
    pub display_index: usize,
    pub record_all: bool,
    pub combine_all: bool,
    pub flip_vertical: bool,
    pub video_bitrate_kbps: u32,
    pub scale_max_width: Option<u32>,
    pub include_audio: bool,
    pub audio_bitrate_kbps: u32,
    pub audio_source: AudioSource, // ADD THIS
}

pub struct Recorder {
    cfg: RecorderConfig,
    stop: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    Avi,
    Webm,
    Mp4,
}

fn flip_frame_vertically(frame: &[u8], width: u32, height: u32) -> Vec<u8> {
    let stride = width as usize * 4;
    let mut flipped_frame = Vec::with_capacity(frame.len());
    for y in (0..height as usize).rev() {
        let start = y * stride;
        let end = start + stride;
        flipped_frame.extend_from_slice(&frame[start..end]);
    }
    flipped_frame
}

impl Recorder {
    pub fn new(cfg: RecorderConfig) -> Self {
        Self {
            cfg,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    pub fn run_blocking(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.cfg.output_dir)?;

        if self.cfg.record_all && self.cfg.combine_all {
            record_combined_all(&self.cfg, &self.stop)?;
            return Ok(());
        }

        if self.cfg.record_all {
            let displays = Screen::all()?;
            if displays.is_empty() {
                return Err(anyhow::anyhow!("no displays found"));
            }
            let mut handles = Vec::with_capacity(displays.len());
            for d_idx in 0..displays.len() {
                let stop_d = self.stop.clone();
                let cfg = self.cfg.clone();
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("recorder_d{}", d_idx))
                        .spawn(move || {
                            let shots = Screenshotter::new(ScreenshotterConfig {
                                output_dir: cfg
                                    .output_dir
                                    .join("screenshots")
                                    .join(format!("display_{}", d_idx)),
                                interval: cfg.screenshot_interval,
                                display_index: d_idx,
                                flip_vertical: cfg.flip_vertical,
                            });
                            let stop_for_shot = stop_d.clone();
                            let _st = std::thread::Builder::new()
                                .name(format!("screenshotter_d{}", d_idx))
                                .spawn(move || {
                                    let _ = shots.run_blocking(&stop_for_shot);
                                });
                            if let Err(err) = record_one_display(&cfg, d_idx, &stop_d) {
                                warn!("display {} recorder exited with error: {:?}", d_idx, err);
                            }
                        })?,
                );
            }
            while !self.stop.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(100));
            }
            for h in handles {
                let _ = h.join();
            }
            return Ok(());
        }

        let screenshotter = Screenshotter::new(ScreenshotterConfig {
            output_dir: self.cfg.output_dir.join("screenshots"),
            interval: self.cfg.screenshot_interval,
            display_index: self.cfg.display_index,
            flip_vertical: self.cfg.flip_vertical,
        });
        let stop_s = self.stop.clone();
        let _t = std::thread::Builder::new()
            .name("screenshotter".into())
            .spawn(move || {
                let _ = screenshotter.run_blocking(&stop_s);
            })?;

        let displays = Screen::all()?;
        if displays.is_empty() {
            return Err(anyhow::anyhow!("no displays found"));
        }
        let idx = self.cfg.display_index.min(displays.len().saturating_sub(1));
        let screen = &displays[idx];
        let probe = screen.capture()?;
        let mut width = probe.width();
        let mut height = probe.height();
        if let Some(max_w) = self.cfg.scale_max_width {
            if max_w > 0 && width > max_w {
                height = (height as u64 * max_w as u64 / width as u64) as u32;
                width = max_w;
            }
        }
        log::info!(
            "container={:?} fps={} size={}x{}",
            self.cfg.container,
            self.cfg.fps,
            width,
            height
        );

        enum WriterKind {
            Avi(AviSegmentWriter),
            #[cfg(feature = "webm")]
            Webm(WebmSegmentWriter),
            Mp4(Mp4SegmentWriter),
        }
        let mut writer = match self.cfg.container {
            Container::Avi => WriterKind::Avi(AviSegmentWriter::create_new(AviSegmentConfig {
                width,
                height,
                fps: self.cfg.fps,
                output_dir: self.cfg.output_dir.join("videos"),
                base_name: self.cfg.base_name.clone(),
            })?),
            Container::Webm => {
                #[cfg(feature = "webm")]
                {
                    WriterKind::Webm(WebmSegmentWriter::create_new(WebmSegmentConfig {
                        width,
                        height,
                        fps: self.cfg.fps,
                        output_dir: self.cfg.output_dir.join("videos"),
                        base_name: self.cfg.base_name.clone(),
                        quantizer: 160,
                    })?)
                }
                #[cfg(not(feature = "webm"))]
                {
                    return Err(anyhow::anyhow!("WebM feature not enabled."));
                }
            }
            Container::Mp4 => {
                let mut w = Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                    width,
                    height,
                    fps: self.cfg.fps,
                    output_dir: self.cfg.output_dir.join("videos"),
                    base_name: self.cfg.base_name.clone(),
                    bitrate_kbps: self.cfg.video_bitrate_kbps,
                    include_audio: self.cfg.include_audio,
                    audio_bitrate_kbps: self.cfg.audio_bitrate_kbps,
                    audio_source: self.cfg.audio_source, // ADD THIS
                })?;
                // Remove the manual start_mic_capture call - it's now automatic
                WriterKind::Mp4(w)
            }
        };
        let mut segment_start = Instant::now();
        let expected_frames_per_segment: u64 =
            (self.cfg.fps as u64).saturating_mul(self.cfg.segment_duration.as_secs());
        let mut frames_in_segment: u64 = 0;

        // ================== REPLACEMENT START ==================
        let frame_duration_100ns = 10_000_000u64 / (self.cfg.fps.max(1) as u64);
        let mut next_frame_time = Instant::now();

        log::info!("Starting video recording loop...");
        while !self.stop.load(Ordering::Relaxed) {
            let now = Instant::now();

            // Check for segment rotation
            if now.duration_since(segment_start) >= self.cfg.segment_duration {
                // your existing segment rotation code
                match &mut writer {
                    WriterKind::Avi(w) => {
                        if let Err(err) = std::mem::replace(
                            w,
                            AviSegmentWriter::create_new(AviSegmentConfig {
                                width,
                                height,
                                fps: self.cfg.fps,
                                output_dir: self.cfg.output_dir.join("videos"),
                                base_name: self.cfg.base_name.clone(),
                            })?,
                        )
                        .finalize()
                        {
                            error!("segment finalize error: {err:?}");
                        }
                    }
                    #[cfg(feature = "webm")]
                    WriterKind::Webm(w) => {
                        if let Err(err) = std::mem::replace(
                            w,
                            WebmSegmentWriter::create_new(WebmSegmentConfig {
                                width,
                                height,
                                fps: self.cfg.fps,
                                output_dir: self.cfg.output_dir.join("videos"),
                                base_name: self.cfg.base_name.clone(),
                                quantizer: 160,
                            })?,
                        )
                        .finalize()
                        {
                            error!("segment finalize error: {err:?}");
                        }
                    }
                    WriterKind::Mp4(w) => {
                        if let Err(err) = std::mem::replace(w, {
                            let nw = Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                                width,
                                height,
                                fps: self.cfg.fps,
                                output_dir: self.cfg.output_dir.join("videos"),
                                base_name: self.cfg.base_name.clone(),
                                bitrate_kbps: self.cfg.video_bitrate_kbps,
                                include_audio: self.cfg.include_audio,
                                audio_bitrate_kbps: self.cfg.audio_bitrate_kbps,
                                audio_source: self.cfg.audio_source, // ADD THIS
                            })?;
                            // Remove manual start_mic_capture
                            nw
                        })
                        .finalize()
                        {
                            error!("segment finalize error: {err:?}");
                        }
                    }
                }
                segment_start = Instant::now();
                frames_in_segment = 0;
            }

            // Wait until it's time for the next frame
            if now < next_frame_time {
                let sleep_duration = next_frame_time.duration_since(now);
                if sleep_duration > Duration::from_millis(1) {
                    std::thread::sleep(sleep_duration);
                }
                continue;
            }

            log::info!("Attempting to capture a frame...");
            let frame = match screen.capture() {
                Ok(f) => {
                    log::info!("Frame captured successfully.");
                    f
                }
                Err(err) => {
                    warn!("capture failed: {err:?}");
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
            };

            // your existing frame processing code
            let src_w = frame.width();
            let src_h = frame.height();
            let bgra = frame.into_raw();
            let bgra_flipped = flip_frame_vertically(&bgra, src_w, src_h);
            let mut rgb_full = Vec::with_capacity((src_w * src_h * 3) as usize);
            rgb_full.extend(
                bgra_flipped
                    .chunks_exact(4)
                    .flat_map(|p| [p[2], p[1], p[0]]),
            );

            let rgb_buf = if src_w == width && src_h == height {
                rgb_full
            } else {
                let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(src_w, src_h, rgb_full)
                    .unwrap();
                image::imageops::resize(&img, width, height, image::imageops::FilterType::Triangle)
                    .into_raw()
            };

            match &mut writer {
                WriterKind::Avi(w) => {
                    let mut jpeg_buf: Vec<u8> = Vec::with_capacity((width * height / 10) as usize);
                    let mut enc = JpegEncoder::new_with_quality(&mut jpeg_buf, 70);
                    if let Err(err) = enc.encode(&rgb_buf, width, height, ColorType::Rgb8.into()) {
                        warn!("jpeg encode failed: {err:?}");
                    }
                    if !jpeg_buf.is_empty() && w.write_jpeg_frame(&jpeg_buf).is_ok() {
                        frames_in_segment += 1;
                        log::info!(
                            "AVI frame write successful. Total frames in segment: {}",
                            frames_in_segment
                        );
                    }
                }
                #[cfg(feature = "webm")]
                WriterKind::Webm(w) => {
                    if w.encode_rgb_frame(&rgb_buf).is_ok() {
                        frames_in_segment += 1;
                    }
                }
                WriterKind::Mp4(w) => {
                    if w.encode_rgb_frame(&rgb_buf).is_ok() {
                        frames_in_segment += 1;
                    }
                }
            }

            // Schedule next frame
            next_frame_time += Duration::from_micros((frame_duration_100ns * 10) / 100);
        }
        // ================== REPLACEMENT END ==================

        match writer {
            WriterKind::Avi(w) => {
                if let Ok(path) = w.finalize() {
                    if frames_in_segment < expected_frames_per_segment {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            #[cfg(feature = "webm")]
            WriterKind::Webm(w) => {
                if let Ok(path) = w.finalize() {
                    if frames_in_segment < expected_frames_per_segment {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            WriterKind::Mp4(w) => {
                if let Ok(path) = w.finalize() {
                    if frames_in_segment < expected_frames_per_segment {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
        Ok(())
    }
}

fn flip_rgb_vertical(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let row = w * 3;
    let mut out = vec![0u8; rgb.len()];
    for j in 0..h {
        let src = j * row;
        let dst = (h - 1 - j) * row;
        out[dst..dst + row].copy_from_slice(&rgb[src..src + row]);
    }
    out
}

fn record_one_display(
    cfg: &RecorderConfig,
    display_index: usize,
    stop: &AtomicBool,
) -> anyhow::Result<()> {
    let displays = Screen::all()?;
    if displays.is_empty() {
        return Err(anyhow::anyhow!("no displays found"));
    }
    let idx = display_index.min(displays.len().saturating_sub(1));
    let screen = &displays[idx];
    let probe = screen.capture()?;
    let width = probe.width();
    let height = probe.height();
    log::info!(
        "display={} container={:?} fps={} size={}x{}",
        idx,
        cfg.container,
        cfg.fps,
        width,
        height
    );
    let frame_interval = Duration::from_millis((1000 / cfg.fps.max(1)) as u64);

    enum WriterKind {
        Avi(AviSegmentWriter),
        #[cfg(feature = "webm")]
        Webm(WebmSegmentWriter),
        Mp4(Mp4SegmentWriter),
    }
    let mut writer = match cfg.container {
        Container::Avi => WriterKind::Avi(AviSegmentWriter::create_new(AviSegmentConfig {
            width,
            height,
            fps: cfg.fps,
            output_dir: cfg
                .output_dir
                .join("videos")
                .join(format!("display_{}", idx)),
            base_name: cfg.base_name.clone(),
        })?),
        Container::Webm => {
            #[cfg(feature = "webm")]
            {
                WriterKind::Webm(WebmSegmentWriter::create_new(WebmSegmentConfig {
                    width,
                    height,
                    fps: cfg.fps,
                    output_dir: cfg
                        .output_dir
                        .join("videos")
                        .join(format!("display_{}", idx)),
                    base_name: cfg.base_name.clone(),
                    quantizer: 160,
                })?)
            }
            #[cfg(not(feature = "webm"))]
            {
                return Err(anyhow::anyhow!("WebM feature not enabled."));
            }
        }
        Container::Mp4 => {
            let w = Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                width,
                height,
                fps: cfg.fps,
                output_dir: cfg
                    .output_dir
                    .join("videos")
                    .join(format!("display_{}", idx)),
                base_name: cfg.base_name.clone(),
                bitrate_kbps: cfg.video_bitrate_kbps,
                include_audio: cfg.include_audio,
                audio_bitrate_kbps: cfg.audio_bitrate_kbps,
                audio_source: cfg.audio_source,
            })?;
            WriterKind::Mp4(w)
        }
    };
    let mut segment_start = Instant::now();
    let expected_frames_per_segment: u64 =
        (cfg.fps as u64).saturating_mul(cfg.segment_duration.as_secs());
    let mut frames_in_segment: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        let now = Instant::now();
        if now.duration_since(segment_start) >= cfg.segment_duration {
            match &mut writer {
                WriterKind::Avi(w) => {
                    if let Err(err) = std::mem::replace(
                        w,
                        AviSegmentWriter::create_new(AviSegmentConfig {
                            width,
                            height,
                            fps: cfg.fps,
                            output_dir: cfg
                                .output_dir
                                .join("videos")
                                .join(format!("display_{}", idx)),
                            base_name: cfg.base_name.clone(),
                        })?,
                    )
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
                #[cfg(feature = "webm")]
                WriterKind::Webm(w) => {
                    if let Err(err) = std::mem::replace(
                        w,
                        WebmSegmentWriter::create_new(WebmSegmentConfig {
                            width,
                            height,
                            fps: cfg.fps,
                            output_dir: cfg
                                .output_dir
                                .join("videos")
                                .join(format!("display_{}", idx)),
                            base_name: cfg.base_name.clone(),
                            quantizer: 160,
                        })?,
                    )
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
                WriterKind::Mp4(w) => {
                    if let Err(err) = std::mem::replace(w, {
                        Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                            width,
                            height,
                            fps: cfg.fps,
                            output_dir: cfg
                                .output_dir
                                .join("videos")
                                .join(format!("display_{}", idx)),
                            base_name: cfg.base_name.clone(),
                            bitrate_kbps: cfg.video_bitrate_kbps,
                            include_audio: cfg.include_audio,
                            audio_bitrate_kbps: cfg.audio_bitrate_kbps,
                            audio_source: cfg.audio_source,
                        })?
                    })
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
            }
            segment_start = Instant::now();
            frames_in_segment = 0;
        }

        let frame = match screen.capture() {
            Ok(f) => f,
            Err(err) => {
                warn!("capture failed: {err:?}");
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        let bgra = frame.into_raw();
        let bgra_flipped = flip_frame_vertically(&bgra, width, height);
        let mut rgb_buf = Vec::with_capacity((width * height * 3) as usize);
        rgb_buf.extend(
            bgra_flipped
                .chunks_exact(4)
                .flat_map(|p| [p[2], p[1], p[0]]),
        );

        match &mut writer {
            WriterKind::Avi(w) => {
                let mut jpeg_buf: Vec<u8> = Vec::with_capacity((width * height / 10) as usize);
                let mut enc = JpegEncoder::new_with_quality(&mut jpeg_buf, 70);
                if let Err(err) = enc.encode(&rgb_buf, width, height, ColorType::Rgb8.into()) {
                    warn!("jpeg encode failed: {err:?}");
                } else if let Err(err) = w.write_jpeg_frame(&jpeg_buf) {
                    warn!("write frame failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
            #[cfg(feature = "webm")]
            WriterKind::Webm(w) => {
                if let Err(err) = w.encode_rgb_frame(&rgb_buf) {
                    warn!("webm encode failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
            WriterKind::Mp4(w) => {
                if let Err(err) = w.encode_rgb_frame(&rgb_buf) {
                    warn!("mp4 encode failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
        }
        std::thread::sleep(frame_interval);
    }

    match writer {
        WriterKind::Avi(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        #[cfg(feature = "webm")]
        WriterKind::Webm(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        WriterKind::Mp4(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
    Ok(())
}

fn record_combined_all(cfg: &RecorderConfig, stop: &AtomicBool) -> anyhow::Result<()> {
    let displays = Screen::all()?;
    if displays.is_empty() {
        return Err(anyhow::anyhow!("no displays found"));
    }
    let mut widths: Vec<u32> = Vec::with_capacity(displays.len());
    let mut heights: Vec<u32> = Vec::with_capacity(displays.len());
    for d in &displays {
        let probe = d.capture()?;
        widths.push(probe.width());
        heights.push(probe.height());
    }
    let total_width: u32 = widths.iter().sum();
    let max_height: u32 = *heights.iter().max().unwrap_or(&0);
    log::info!(
        "combine_all: displays={} size={}x{}",
        displays.len(),
        total_width,
        max_height
    );
    let frame_interval = Duration::from_millis((1000 / cfg.fps.max(1)) as u64);

    enum WriterKind {
        Avi(AviSegmentWriter),
        #[cfg(feature = "webm")]
        Webm(WebmSegmentWriter),
        Mp4(Mp4SegmentWriter),
    }
    let mut writer = match cfg.container {
        Container::Avi => WriterKind::Avi(AviSegmentWriter::create_new(AviSegmentConfig {
            width: total_width,
            height: max_height,
            fps: cfg.fps,
            output_dir: cfg.output_dir.join("videos"),
            base_name: cfg.base_name.clone(),
        })?),
        Container::Webm => {
            #[cfg(feature = "webm")]
            {
                WriterKind::Webm(WebmSegmentWriter::create_new(WebmSegmentConfig {
                    width: total_width,
                    height: max_height,
                    fps: cfg.fps,
                    output_dir: cfg.output_dir.join("videos"),
                    base_name: cfg.base_name.clone(),
                    quantizer: 160,
                })?)
            }
            #[cfg(not(feature = "webm"))]
            {
                return Err(anyhow::anyhow!("WebM feature not enabled."));
            }
        }
        Container::Mp4 => {
            let w = Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                width: total_width,
                height: max_height,
                fps: cfg.fps,
                output_dir: cfg.output_dir.join("videos"),
                base_name: cfg.base_name.clone(),
                bitrate_kbps: cfg.video_bitrate_kbps,
                include_audio: cfg.include_audio,
                audio_bitrate_kbps: cfg.audio_bitrate_kbps,
                audio_source: cfg.audio_source,
            })?;
            WriterKind::Mp4(w)
        }
    };
    let mut segment_start = Instant::now();
    let expected_frames_per_segment: u64 =
        (cfg.fps as u64).saturating_mul(cfg.segment_duration.as_secs());
    let mut frames_in_segment: u64 = 0;
    let mut last_shot = Instant::now() - cfg.screenshot_interval;

    let mut rgb_out: Vec<u8> = vec![0u8; (total_width * max_height * 3) as usize];
    let mut x_offsets: Vec<u32> = Vec::with_capacity(displays.len());
    {
        let mut acc = 0u32;
        for &w in &widths {
            x_offsets.push(acc);
            acc = acc.saturating_add(w);
        }
    }

    while !stop.load(Ordering::Relaxed) {
        let now = Instant::now();
        if now.duration_since(segment_start) >= cfg.segment_duration {
            match &mut writer {
                WriterKind::Avi(w) => {
                    if let Err(err) = std::mem::replace(
                        w,
                        AviSegmentWriter::create_new(AviSegmentConfig {
                            width: total_width,
                            height: max_height,
                            fps: cfg.fps,
                            output_dir: cfg.output_dir.join("videos"),
                            base_name: cfg.base_name.clone(),
                        })?,
                    )
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
                #[cfg(feature = "webm")]
                WriterKind::Webm(w) => {
                    if let Err(err) = std::mem::replace(
                        w,
                        WebmSegmentWriter::create_new(WebmSegmentConfig {
                            width: total_width,
                            height: max_height,
                            fps: cfg.fps,
                            output_dir: cfg.output_dir.join("videos"),
                            base_name: cfg.base_name.clone(),
                            quantizer: 160,
                        })?,
                    )
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
                WriterKind::Mp4(w) => {
                    if let Err(err) = std::mem::replace(w, {
                        Mp4SegmentWriter::create_new(Mp4SegmentConfig {
                            width: total_width,
                            height: max_height,
                            fps: cfg.fps,
                            output_dir: cfg.output_dir.join("videos"),
                            base_name: cfg.base_name.clone(),
                            bitrate_kbps: cfg.video_bitrate_kbps,
                            include_audio: cfg.include_audio,
                            audio_bitrate_kbps: cfg.audio_bitrate_kbps,
                            audio_source: cfg.audio_source,
                        })?
                    })
                    .finalize()
                    {
                        error!("segment finalize error: {err:?}");
                    }
                }
            }
            segment_start = Instant::now();
            frames_in_segment = 0;
        }

        rgb_out.fill(0);
        for (i, screen) in displays.iter().enumerate() {
            let frame = match screen.capture() {
                Ok(f) => f,
                Err(err) => {
                    warn!("capture failed on display {}: {:?}", i, err);
                    continue;
                }
            };
            let w = widths[i] as usize;
            let h = heights[i] as usize;
            let x_off = x_offsets[i] as usize;
            let bgra = frame.into_raw();
            let bgra_flipped = flip_frame_vertically(&bgra, w as u32, h as u32);
            let mut rgb_buf = Vec::with_capacity(w * h * 3);
            rgb_buf.extend(
                bgra_flipped
                    .chunks_exact(4)
                    .flat_map(|p| [p[2], p[1], p[0]]),
            );
            for y in 0..h {
                let out_row_start = (y * (total_width as usize) + x_off) * 3;
                let src_row_start = y * w * 3;
                let copy_len = w * 3;
                rgb_out[out_row_start..out_row_start + copy_len]
                    .copy_from_slice(&rgb_buf[src_row_start..src_row_start + copy_len]);
            }
        }

        match &mut writer {
            WriterKind::Avi(w) => {
                let mut jpeg_buf: Vec<u8> =
                    Vec::with_capacity((total_width * max_height / 10) as usize);
                let mut enc = JpegEncoder::new_with_quality(&mut jpeg_buf, 70);
                let rgb_enc = rgb_out.clone();
                if let Err(err) =
                    enc.encode(&rgb_enc, total_width, max_height, ColorType::Rgb8.into())
                {
                    warn!("jpeg encode failed: {err:?}");
                } else if let Err(err) = w.write_jpeg_frame(&jpeg_buf) {
                    warn!("write frame failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
            #[cfg(feature = "webm")]
            WriterKind::Webm(w) => {
                let rgb_enc = rgb_out.clone();
                if let Err(err) = w.encode_rgb_frame(&rgb_enc) {
                    warn!("webm encode failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
            WriterKind::Mp4(w) => {
                let rgb_enc = rgb_out.clone();
                if let Err(err) = w.encode_rgb_frame(&rgb_enc) {
                    warn!("mp4 encode failed: {err:?}");
                } else {
                    frames_in_segment += 1;
                }
            }
        }

        if last_shot.elapsed() >= cfg.screenshot_interval {
            if let Err(err) = (|| -> anyhow::Result<()> {
                let dir = cfg.output_dir.join("screenshots").join("combined");
                std::fs::create_dir_all(&dir)?;
                let now: DateTime<Local> = Local::now();
                let path = dir.join(format!("screenshot_{}.jpg", now.format("%Y%m%d_%H%M%S")));
                let mut file = File::create(&path)?;
                let mut enc = JpegEncoder::new_with_quality(&mut file, 80);
                enc.encode(&rgb_out, total_width, max_height, ColorType::Rgb8.into())?;
                Ok(())
            })() {
                warn!("combined screenshot save failed: {:?}", err);
            } else {
                last_shot = Instant::now();
            }
        }
        std::thread::sleep(frame_interval);
    }

    match writer {
        WriterKind::Avi(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        #[cfg(feature = "webm")]
        WriterKind::Webm(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        WriterKind::Mp4(w) => {
            if let Ok(path) = w.finalize() {
                if frames_in_segment < expected_frames_per_segment {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }
    Ok(())
}
