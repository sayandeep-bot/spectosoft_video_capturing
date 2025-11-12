use byteorder::{LittleEndian, WriteBytesExt};
use chrono::{DateTime, Local};
use log::info;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AviSegmentConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub output_dir: PathBuf,
    pub base_name: String, // e.g., "recording"
}

#[derive(Debug, Clone, Copy)]
struct IdxEntry {
    // FourCC of chunk, always '00dc' for video frame
    ckid: [u8; 4],
    flags: u32,
    offset_from_movi: u32,
    length: u32,
}

pub struct AviSegmentWriter {
    _cfg: AviSegmentConfig,
    file: File,
    // positions for later size fixups
    riff_size_pos: u64,
    _movi_list_start: u64,
    movi_size_pos: u64,
    idx: Vec<IdxEntry>,
    frames_written: u32,
    max_frame_size: u32,
    _start_time: DateTime<Local>,
    pub output_path: PathBuf,
}

impl AviSegmentWriter {
    pub fn create_new(cfg: AviSegmentConfig) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&cfg.output_dir)?;
        let now: DateTime<Local> = Local::now();
        let filename = format!(
            "{}_{}.avi",
            cfg.base_name,
            now.format("%Y%m%d_%H%M%S")
        );
        let path = cfg.output_dir.join(filename);

        let mut file = File::create(&path)?;

        // RIFF header
        file.write_all(b"RIFF")?;
        let riff_size_pos = file.stream_position()?; // placeholder for RIFF size
        file.write_u32::<LittleEndian>(0)?; // will fill later: file_size - 8
        file.write_all(b"AVI ")?;

        // LIST 'hdrl'
        let mut hdrl_buf: Vec<u8> = Vec::with_capacity(2048);
        let suggested_buf = cfg.width.saturating_mul(cfg.height).saturating_mul(3);
        write_avih(&mut hdrl_buf, cfg.width, cfg.height, cfg.fps, 0, suggested_buf)?; // total frames unknown at start
        write_strl(&mut hdrl_buf, cfg.width, cfg.height, cfg.fps, suggested_buf)?;

        // write LIST header for hdrl
        file.write_all(b"LIST")?;
        let hdrl_size = 4 + hdrl_buf.len() as u32; // includes 'hdrl'
        file.write_u32::<LittleEndian>(hdrl_size)?;
        file.write_all(b"hdrl")?;
        file.write_all(&hdrl_buf)?;

        // LIST 'movi'
        file.write_all(b"LIST")?;
        let movi_size_pos = file.stream_position()?;
        file.write_u32::<LittleEndian>(0)?; // placeholder size
        file.write_all(b"movi")?;
        let movi_list_start = movi_size_pos - 4; // points to size field; list starts at 'LIST'

        Ok(Self {
            _cfg: cfg,
            file,
            riff_size_pos,
            _movi_list_start: movi_list_start,
            movi_size_pos,
            idx: Vec::with_capacity(4096),
            frames_written: 0,
            max_frame_size: 0,
            _start_time: now,
            output_path: path,
        })
    }

    pub fn write_jpeg_frame(&mut self, jpeg_bytes: &[u8]) -> anyhow::Result<()> {
        // chunk: '00dc' + size + data (+ pad)
        let chunk_id = *b"00dc";
        let chunk_id_pos = self.file.stream_position()?; // position at '00dc'
        self.file.write_all(&chunk_id)?;
        self.file
            .write_u32::<LittleEndian>(jpeg_bytes.len() as u32)?;
        let chunk_data_start = self.file.stream_position()?; // position after size
        self.file.write_all(jpeg_bytes)?;
        // pad to even size
        if jpeg_bytes.len() % 2 != 0 {
            self.file.write_all(&[0x00])?;
        }
        let _chunk_end = self.file.stream_position()?;
        let _chunk_len_padded = (_chunk_end - chunk_data_start) as u32;

        // Index offset is from start of movi list data (after 'movi' fourcc), per AVI spec
        let movi_data_start = self.movi_size_pos + 4; // after 'movi'
        let offset_from_movi = (chunk_id_pos - movi_data_start) as u32; // offset to '00dc'

        self.idx.push(IdxEntry {
            ckid: chunk_id,
            flags: 0x10, // AVIIF_KEYFRAME
            offset_from_movi,
            length: jpeg_bytes.len() as u32,
        });
        self.frames_written = self.frames_written.saturating_add(1);
        self.max_frame_size = self.max_frame_size.max(jpeg_bytes.len() as u32);
        Ok(())
    }

    pub fn finalize(mut self) -> anyhow::Result<PathBuf> {
        // If no frames written, delete the file to avoid zero/tiny segments
        if self.frames_written == 0 {
            let path = self.output_path.clone();
            // Ensure file handle is dropped before deletion
            drop(self.file);
            let _ = std::fs::remove_file(&path);
            info!("avi segment had no frames, removed: {}", path.display());
            return Ok(path);
        }
        // finalize movi size
        let file_len = self.file.seek(SeekFrom::End(0))?;
        let movi_size = (file_len - (self.movi_size_pos + 4)) as u32; // includes 'movi' + chunks
        let cur = self.file.stream_position()?;
        self.file.seek(SeekFrom::Start(self.movi_size_pos))?;
        self.file.write_u32::<LittleEndian>(movi_size)?;
        self.file.seek(SeekFrom::Start(cur))?;

        // write idx1
        self.file.write_all(b"idx1")?;
        let idx_size = (self.idx.len() as u32) * 16;
        self.file.write_u32::<LittleEndian>(idx_size)?;
        for e in &self.idx {
            self.file.write_all(&e.ckid)?;
            self.file.write_u32::<LittleEndian>(e.flags)?;
            self.file.write_u32::<LittleEndian>(e.offset_from_movi)?;
            self.file.write_u32::<LittleEndian>(e.length)?;
        }

        // fix RIFF size
        let final_len = self.file.seek(SeekFrom::End(0))?;
        let riff_size = (final_len - 8) as u32;
        self.file.seek(SeekFrom::Start(self.riff_size_pos))?;
        self.file.write_u32::<LittleEndian>(riff_size)?;
        self.file.flush()?;

        info!(
            "avi segment finalized: {} frames -> {}",
            self.frames_written,
            self.output_path.display()
        );
        Ok(self.output_path)
    }
}

fn write_avih<W: Write>(
    mut w: W,
    width: u32,
    height: u32,
    fps: u32,
    total_frames: u32,
    suggested_buf: u32,
) -> anyhow::Result<()> {
    let usec_per_frame = 1_000_000u32 / fps.max(1);
    // 'avih'
    w.write_all(b"avih")?;
    w.write_u32::<LittleEndian>(56)?; // size of AVIMAINHEADER
    w.write_u32::<LittleEndian>(usec_per_frame)?; // dwMicroSecPerFrame
    w.write_u32::<LittleEndian>(0)?; // dwMaxBytesPerSec (0 ok)
    w.write_u32::<LittleEndian>(0)?; // dwPaddingGranularity
    w.write_u32::<LittleEndian>(0x10)?; // dwFlags: HAS_INDEX
    w.write_u32::<LittleEndian>(total_frames)?; // dwTotalFrames (0 ok before we know)
    w.write_u32::<LittleEndian>(0)?; // dwInitialFrames
    w.write_u32::<LittleEndian>(1)?; // dwStreams
    w.write_u32::<LittleEndian>(suggested_buf)?; // dwSuggestedBufferSize
    w.write_u32::<LittleEndian>(width)?; // dwWidth
    w.write_u32::<LittleEndian>(height)?; // dwHeight
    for _ in 0..4 {
        w.write_u32::<LittleEndian>(0)?; // dwReserved[4]
    }
    Ok(())
}

fn write_strl<W: Write>(mut w: W, width: u32, height: u32, fps: u32, suggested_buf: u32) -> anyhow::Result<()> {
    // Build 'strl' list in a buffer to compute size
    let mut strl: Vec<u8> = Vec::with_capacity(512);
    // 'strh' - stream header
    strl.write_all(b"strh")?;
    strl.write_u32::<LittleEndian>(56)?; // size of AVISTREAMHEADER
    strl.write_all(b"vids")?; // fccType
    strl.write_all(b"MJPG")?; // fccHandler
    strl.write_u32::<LittleEndian>(0)?; // dwFlags
    strl.write_u16::<LittleEndian>(0)?; // wPriority
    strl.write_u16::<LittleEndian>(0)?; // wLanguage
    strl.write_u32::<LittleEndian>(0)?; // dwInitialFrames
    strl.write_u32::<LittleEndian>(1)?; // dwScale
    strl.write_u32::<LittleEndian>(fps.max(1))?; // dwRate
    strl.write_u32::<LittleEndian>(0)?; // dwStart
    strl.write_u32::<LittleEndian>(0)?; // dwLength (0 unknown)
    strl.write_u32::<LittleEndian>(suggested_buf)?; // dwSuggestedBufferSize
    strl.write_u32::<LittleEndian>(-1i32 as u32)?; // dwQuality (-1 default)
    strl.write_u32::<LittleEndian>(0)?; // dwSampleSize (0 for variable)
    // rcFrame
    strl.write_u16::<LittleEndian>(0)?; // left
    strl.write_u16::<LittleEndian>(0)?; // top
    strl.write_u16::<LittleEndian>(width as u16)?; // right
    strl.write_u16::<LittleEndian>(height as u16)?; // bottom

    // 'strf' - stream format (BITMAPINFOHEADER)
    strl.write_all(b"strf")?;
    strl.write_u32::<LittleEndian>(40)?; // biSize
    strl.write_u32::<LittleEndian>(width)?; // biWidth
    strl.write_u32::<LittleEndian>(height)?; // biHeight
    strl.write_u16::<LittleEndian>(1)?; // biPlanes
    strl.write_u16::<LittleEndian>(24)?; // biBitCount (24 is common for MJPEG)
    strl.write_all(b"MJPG")?; // biCompression
    strl.write_u32::<LittleEndian>(0)?; // biSizeImage (0 ok for MJPEG)
    strl.write_u32::<LittleEndian>(0)?; // biXPelsPerMeter
    strl.write_u32::<LittleEndian>(0)?; // biYPelsPerMeter
    strl.write_u32::<LittleEndian>(0)?; // biClrUsed
    strl.write_u32::<LittleEndian>(0)?; // biClrImportant

    // write LIST header
    w.write_all(b"LIST")?;
    let strl_size = 4 + strl.len() as u32; // includes 'strl'
    w.write_u32::<LittleEndian>(strl_size)?;
    w.write_all(b"strl")?;
    w.write_all(&strl)?;
    Ok(())
}


