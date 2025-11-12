use chrono::{DateTime, Local};
use log::{info, warn};
use rav1e::prelude::*;
// These imports are now correct based on your new code structure.
use webm::mux::{Segment, SegmentBuilder, VideoCodecId, VideoTrack, Writer};
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct WebmSegmentConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output_dir: PathBuf,
    pub base_name: String,
    pub quantizer: usize,
}

pub struct WebmSegmentWriter {
    // This now correctly holds a Segment, not a Muxer.
    segment: Segment<BufWriter<File>>,
    video_track: VideoTrack,
    encoder: Context<u8>,
    frame_count: u64,
    fps: u64,
    width: usize,
    height: usize,
    output_path: PathBuf,
}

impl WebmSegmentWriter {
    pub fn create_new(cfg: WebmSegmentConfig) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&cfg.output_dir)?;
        let now: DateTime<Local> = Local::now();
        let filename = format!("{}_{}.webm", cfg.base_name, now.format("%Y%m%d_%H%M%S"));
        let path = cfg.output_dir.join(filename);
        let file = File::create(&path)?;
        let writer = Writer::new(BufWriter::new(file));

        let builder = SegmentBuilder::new(writer)?;
        let (builder, video_track) = builder.add_video_track(
            cfg.width,
            cfg.height,
            VideoCodecId::AV1,
            None,
        )?;
        let segment = builder.build();

        let mut enc_config = EncoderConfig::with_speed_preset(10);
        enc_config.width = cfg.width as usize;
        enc_config.height = cfg.height as usize;
        enc_config.quantizer = cfg.quantizer.min(255);
        enc_config.speed_settings.rdo_lookahead_frames = 1;
        enc_config.time_base = Rational::new(1, cfg.fps as u64);

        let rav1e_cfg = Config::new()
            .with_encoder_config(enc_config)
            .with_threads(4);
        let encoder: Context<u8> = rav1e_cfg.new_context()?;

        Ok(Self {
            segment,
            video_track,
            encoder,
            frame_count: 0,
            fps: cfg.fps as u64,
            width: cfg.width as usize,
            height: cfg.height as usize,
            output_path: path,
        })
    }

    pub fn encode_rgb_frame(&mut self, rgb: &[u8]) -> anyhow::Result<()> {
        let mut frame = self.encoder.new_frame();
        fill_yuv_planes_from_rgb(&mut frame, rgb, self.width, self.height);
        self.encoder.send_frame(frame)?;
        self.receive_packets()?;
        Ok(())
    }

    fn receive_packets(&mut self) -> anyhow::Result<()> {
        loop {
            match self.encoder.receive_packet() {
                Ok(packet) => {
                    let timestamp_ns = (self.frame_count * 1_000_000_000) / self.fps;
                    let is_keyframe = packet.frame_type == FrameType::KEY;
                    
                    self.segment.add_frame(
                        self.video_track,
                        &packet.data,
                        timestamp_ns,
                        is_keyframe,
                    )?;
                    self.frame_count += 1;
                }
                Err(EncoderStatus::LimitReached) => break,
                Err(EncoderStatus::Encoded) | Err(EncoderStatus::NeedMoreData) => break,
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to receive packet from rav1e: {:?}",
                        e
                    ));
                }
            }
        }
        Ok(())
    }

    // Changed the receiver to `self` from `mut self` because finalize consumes `self.segment`
    pub fn finalize(mut self) -> anyhow::Result<PathBuf> {
        loop {
            self.receive_packets()?;
            if self.encoder.send_frame(None).is_err() {
                break;
            }
        }

        // finalize() consumes self.segment, so we can't use it after this.
        if let Err(e) = self.segment.finalize(None) {
            // It's good practice to log the actual error for debugging.
            warn!("Error finalizing webm segment: {:?}", e);
        }

        if self.frame_count == 0 {
            // CORRECTED: The `drop` call is removed. Since `finalize` consumed the segment,
            // the file handle is already closed, and we can safely delete the file.
            let _ = std::fs::remove_file(&self.output_path);
            info!(
                "WebM segment had no frames, removed: {}",
                self.output_path.display()
            );
        } else {
            info!(
                "WebM segment finalized: {} frames -> {}",
                self.frame_count,
                self.output_path.display()
            );
        }

        Ok(self.output_path)
    }
}

fn fill_yuv_planes_from_rgb(frame: &mut Frame<u8>, rgb_data: &[u8], width: usize, height: usize) {
    if let [y_plane, u_plane, v_plane] = frame.planes.as_mut_slice() {
        for row in 0..height {
            let y_stride = y_plane.cfg.stride;
            let uv_stride = u_plane.cfg.stride;

            let y_row = &mut y_plane.data_origin_mut()[(row * y_stride)..];
            let u_row = &mut u_plane.data_origin_mut()[((row / 2) * uv_stride)..];
            let v_row = &mut v_plane.data_origin_mut()[((row / 2) * uv_stride)..];

            for col in 0..width {
                let r = rgb_data[(row * width + col) * 3] as f32;
                let g = rgb_data[(row * width + col) * 3 + 1] as f32;
                let b = rgb_data[(row * width + col) * 3 + 2] as f32;

                let y = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
                let u = (-0.169 * r - 0.331 * g + 0.5 * b + 128.0).round().clamp(0.0, 255.0) as u8;
                let v = (0.5 * r - 0.419 * g - 0.081 * b + 128.0).round().clamp(0.0, 255.0) as u8;

                y_row[col] = y;

                if row % 2 == 0 && col % 2 == 0 {
                    u_row[col / 2] = u;
                    v_row[col / 2] = v;
                }
            }
        }
    }
}