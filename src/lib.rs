mod activity;
mod avi_writer;
mod mp4_writer;
mod recorder;
mod screenshotter;
#[cfg(feature = "webm")]
mod webm_writer;

pub use activity::{ActivityBatcher, ActivityConfig};
pub use avi_writer::{AviSegmentConfig, AviSegmentWriter};
pub use mp4_writer::{AudioSource, Mp4SegmentConfig, Mp4SegmentWriter};
pub use recorder::{Container, Recorder, RecorderConfig};
pub use screenshotter::{Screenshotter, ScreenshotterConfig};
#[cfg(feature = "webm")]
pub use webm_writer::{WebmSegmentConfig, WebmSegmentWriter};
