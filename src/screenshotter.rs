use chrono::{DateTime, Local};
use image::{codecs::jpeg::JpegEncoder, ColorType};
use log::{error, info};
use screenshots::Screen; // cross-platform screenshot
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::fs::File;

#[derive(Debug, Clone)]
pub struct ScreenshotterConfig {
    pub output_dir: PathBuf,
    pub interval: Duration,
    pub display_index: usize,
    pub flip_vertical: bool,
}

pub struct Screenshotter {
    config: ScreenshotterConfig,
}

impl Screenshotter {
    pub fn new(config: ScreenshotterConfig) -> Self {
        Self { config }
    }

    pub fn run_blocking(&self, stop_flag: &std::sync::atomic::AtomicBool) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.config.output_dir)?;

        let mut last = Instant::now() - self.config.interval;
        while !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            if last.elapsed() >= self.config.interval {
                if let Err(err) = self.capture_once() {
                    error!("screenshot capture failed: {err:?}");
                }
                last = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Ok(())
    }

    pub fn capture_once(&self) -> anyhow::Result<PathBuf> {
        let displays = Screen::all()?;
        if displays.is_empty() { return Err(anyhow::anyhow!("no displays found")); }
        let idx = self.config.display_index.min(displays.len().saturating_sub(1));
        if idx != self.config.display_index { log::warn!("requested display_index={} out of range, using {}", self.config.display_index, idx); }
        let screen = &displays[idx];
        let image = screen.capture()?; // BGRA
        let width = image.width() as u32;
        let height = image.height() as u32;

        // Convert BGRA -> RGB (drop alpha) without intermediate swap
        let bgra = image.into_raw();
        let mut rgb_buf = Vec::with_capacity((width * height * 3) as usize);
        rgb_buf.extend(bgra.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]));
        if self.config.flip_vertical {
            // Flip vertically row-wise using split_at_mut to avoid overlapping borrows
            let w = width as usize;
            let h = height as usize;
            let row = w * 3;
            for j in 0..(h / 2) {
                let a = j * row;
                let b = (h - 1 - j) * row;
                let (head, tail) = rgb_buf.split_at_mut(b);
                head[a..a+row].swap_with_slice(&mut tail[..row]);
            }
        }

        let now: DateTime<Local> = Local::now();
        let filename = format!("screenshot_{}.jpg", now.format("%Y%m%d_%H%M%S"));
        let path = self.config.output_dir.join(filename);
        let mut file = File::create(&path)?;
        let mut enc = JpegEncoder::new_with_quality(&mut file, 80);
        enc.encode(&rgb_buf, width, height, ColorType::Rgb8.into())?;
        info!("screenshot saved: {}", path.display());
        Ok(path)
    }
}


