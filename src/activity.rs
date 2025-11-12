use chrono::{DateTime, Utc};
use crossbeam_channel::{Receiver, RecvTimeoutError};
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivityEvent {
    KeyPress(char),
    MouseClick { button: String, x: i32, y: i32 },
    MouseScroll { delta_y: i32, x: i32, y: i32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEntry {
    timestamp: DateTime<Utc>,
    #[serde(flatten)]
    event: ActivityEvent,
}

impl ActivityEntry {
    pub fn new(event: ActivityEvent) -> Self {
        Self {
            timestamp: Utc::now(),
            event,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActivityConfig {
    /// Directory to save the activity logs.
    pub output_dir: PathBuf,
    /// How often to write the batch to a file, even if it's not full.
    pub flush_interval: Duration,
    /// The maximum number of events to hold in memory before forcing a flush.
    pub batch_capacity: usize,
}

/// Listens on a channel for `ActivityEvent`s and batches them to disk.
pub struct ActivityBatcher {
    config: ActivityConfig,
    receiver: Receiver<ActivityEvent>,
}

impl ActivityBatcher {
    pub fn new(config: ActivityConfig, receiver: Receiver<ActivityEvent>) -> Self {
        Self { config, receiver }
    }

    /// Runs the batching loop. This is a blocking operation.
    pub fn run_blocking(&self) -> anyhow::Result<()> {
        create_dir_all(&self.config.output_dir)?;
        let mut batch = Vec::with_capacity(self.config.batch_capacity);

        loop {
            // Block until the first event arrives or timeout occurs
            match self.receiver.recv_timeout(self.config.flush_interval) {
                Ok(event) => {
                    batch.push(ActivityEntry::new(event));
                    // After receiving one event, quickly drain any others that are pending
                    // to fill up the batch faster.
                    while batch.len() < self.config.batch_capacity {
                        if let Ok(event) = self.receiver.try_recv() {
                            batch.push(ActivityEntry::new(event));
                        } else {
                            break; // No more pending events
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Timeout reached, flush whatever we have
                }
                Err(RecvTimeoutError::Disconnected) => {
                    info!("Activity channel disconnected. Flushing remaining events and shutting down.");
                    // If the channel is disconnected, flush one last time and exit.
                    self.flush_batch(&mut batch)?;
                    return Ok(());
                }
            };

            // Flush if the batch is full or if we timed out with a non-empty batch
            if !batch.is_empty() {
                self.flush_batch(&mut batch)?;
            }
        }
    }

    /// Writes the current batch of events to a uniquely named JSON file.
    fn flush_batch(&self, batch: &mut Vec<ActivityEntry>) -> anyhow::Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let filename = format!("activity_{}_{}.json", Utc::now().format("%Y%m%d_%H%M%S"), Uuid::new_v4());
        let path = self.config.output_dir.join(filename);
        
        let mut file = File::create(&path)?;
        let json_string = serde_json::to_string(&batch)?;
        file.write_all(json_string.as_bytes())?;

        info!("Flushed {} activity events to {}", batch.len(), path.display());
        batch.clear();
        Ok(())
    }
}