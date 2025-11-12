use clap::{Parser, Subcommand, ValueEnum};
// VVVV --- FIX: Import AudioSource from your library --- VVVV
use recording_project::{Container, Recorder, RecorderConfig, AudioSource};
use std::path::PathBuf;
use std::process;
// VVVV --- FIX: Add the missing imports for AtomicBool and Ordering --- VVVV
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use fs_extra::{dir, file};

/// A CLI tool to control screen recording.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// VVVV --- FIX: Add the CliAudioSource enum definition --- VVVV
#[derive(ValueEnum, Debug, Clone, Copy)]
enum CliAudioSource {
    Mic,
    System,
    Both,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Starts the recording process in the background.
    Start {
        /// Sets the output directory for videos and screenshots.
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,

        /// Frames per second for the video recording.
        #[arg(long, default_value_t = 15)]
        fps: u32,

        /// Video container format.
        #[arg(long, value_enum, default_value_t = ContainerFormat::Mp4)]
        container: ContainerFormat,

        /// Duration of each video segment in seconds.
        #[arg(long, default_value_t = 60)]
        segment_duration: u64,

        /// Enable audio recording (requires 'audio_capture' feature).
        #[arg(long)]
        audio: bool,

        // VVVV --- FIX: Add the missing `audio_source` field --- VVVV
        /// Select the audio source to record.
        #[arg(long, value_enum, default_value_t = CliAudioSource::Both)]
        audio_source: CliAudioSource,
    },
    /// Stops the currently running recording process.
    Stop,
    /// Checks if the recorder is currently running.
    Status,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum ContainerFormat {
    Avi,
    Webm,
    Mp4,
}

// Helper to manage state files (pid, stop signal)
fn get_state_dir() -> anyhow::Result<PathBuf> {
    let dir = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
        .join(".recording_project");
    dir::create_all(&dir, true)?;
    Ok(dir)
}

fn get_pid_file_path() -> anyhow::Result<PathBuf> {
    Ok(get_state_dir()?.join("recorder.pid"))
}

fn get_stop_file_path() -> anyhow::Result<PathBuf> {
    Ok(get_state_dir()?.join("recorder.stop"))
}

fn main() -> anyhow::Result<()> {
    // Initialize logger for feedback
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let pid_file = get_pid_file_path()?;
    let stop_file = get_stop_file_path()?;

    match cli.command {
        // VVVV --- FIX: The `audio_source` field can now be correctly extracted --- VVVV
        Commands::Start {
            output,
            fps,
            container,
            segment_duration,
            audio,
            audio_source,
        } => {
            if pid_file.exists() {
                log::error!("Recorder is already running. Please use 'stop' first.");
                process::exit(1);
            }

            file::write_all(&pid_file, &process::id().to_string())?;
            log::info!("Recorder started with PID: {}", process::id());

            let running = Arc::new(AtomicBool::new(true));
            let r = running.clone();
            ctrlc::set_handler(move || {
                r.store(false, Ordering::SeqCst);
            })
            .expect("Error setting Ctrl+C handler");

            let output_dir = output.unwrap_or_else(|| dirs::home_dir().unwrap().join("recordings"));

            let recorder_cfg = RecorderConfig {
                output_dir,
                base_name: "recording".to_string(),
                segment_duration: Duration::from_secs(segment_duration),
                fps,
                screenshot_interval: Duration::from_secs(30),
                container: match container {
                    ContainerFormat::Avi => Container::Avi,
                    ContainerFormat::Webm => Container::Webm,
                    ContainerFormat::Mp4 => Container::Mp4,
                },
                display_index: 0,
                record_all: false,
                combine_all: false,
                flip_vertical: false,
                video_bitrate_kbps: 4000,
                scale_max_width: None,
                include_audio: audio,
                audio_bitrate_kbps: 128,
                audio_source: match audio_source {
                    CliAudioSource::Mic => AudioSource::Microphone,
                    CliAudioSource::System => AudioSource::System,
                    CliAudioSource::Both => AudioSource::Both,
                },
            };

            let recorder = Recorder::new(recorder_cfg);
            let stop_flag = recorder.stop_flag();

            let recorder_handle = thread::spawn(move || {
                if let Err(e) = recorder.run_blocking() {
                    log::error!("Recorder thread exited with error: {}", e);
                }
            });

            while running.load(Ordering::SeqCst) {
                if stop_file.exists() {
                    log::info!("Stop signal received.");
                    break;
                }
                thread::sleep(Duration::from_secs(1));
            }

            log::info!("Shutting down recorder gracefully...");
            stop_flag.store(true, Ordering::Relaxed);
            recorder_handle.join().expect("Recorder thread panicked");

            file::remove(&pid_file)?;
            file::remove(&stop_file).ok();
            log::info!("Recorder stopped and cleaned up.");
        }
        Commands::Stop => {
            if !pid_file.exists() {
                log::warn!("Recorder does not appear to be running (no PID file found).");
                return Ok(());
            }
            log::info!("Sending stop signal to the recorder...");
            file::write_all(&stop_file, "")?;
        }
        Commands::Status => {
            if pid_file.exists() {
                let pid = file::read_to_string(&pid_file)?;
                println!("Recorder is running with PID: {}", pid.trim());
            } else {
                println!("Recorder is not running.");
            }
        }
    }

    Ok(())
}
