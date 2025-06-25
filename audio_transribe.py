#!/usr/bin/env python3
"""
Audio Transcriber - Professional Speech-to-Text Tool
====================================================

A command-line tool for transcribing audio files with optional preprocessing
to improve transcription quality. Supports multiple audio formats and provides
extensive customization options.

METADATA
--------
File:        audio_transcribe.py
Author:      Dennis 'dendogg' Smaltz
Version:     5.1.0
Date:        2024-01-15
License:     MIT License

FEATURES
--------
• Multi-format audio support (MP3, WAV, FLAC, OGG, etc.)
• Batch processing with glob patterns and directories
• Optional audio preprocessing for improved accuracy
• Gentle noise reduction and voice optimization
• Multiple Whisper model sizes
• Multiple output formats (TXT, JSON, SRT, VTT)
• Automatic language detection
• Detailed logging and progress tracking
• Memory-efficient processing for large files
• Robust error handling and recovery
• Signal handling for graceful shutdown

REQUIREMENTS
------------
pip install openai-whisper pydub noisereduce numpy scipy soundfile tqdm

EXTERNAL DEPENDENCIES
--------------------
FFmpeg must be installed:
  • Windows: Run the FFmpeg installer script
  • macOS:   brew install ffmpeg
  • Linux:   sudo apt install ffmpeg

BASIC USAGE
-----------
python audio_transcribe.py <audio_files> [options]

EXAMPLES
--------
# Single file transcription
python audio_transcribe.py interview.mp3

# Batch processing with glob pattern
python audio_transcribe.py "audio/*.mp3" --output-dir transcripts/

# Multiple output formats
python audio_transcribe.py recording.wav --format txt,json,srt

# Process entire directory recursively
python audio_transcribe.py --input-dir recordings/ --recursive --format json

# Advanced batch processing
python audio_transcribe.py "audio/*.mp3" --model large --language en --format srt,vtt

KEY OPTIONS
-----------
--input-dir     : Process all audio files in directory
--recursive     : Include subdirectories when using --input-dir
--format        : Output format(s): txt,json,srt,vtt (default: txt)
--model         : Whisper model size (tiny/base/small/medium/large)
--no-preprocess : Skip audio preprocessing
--save-processed: Keep processed audio files
--language      : Force specific language (auto-detect if omitted)
--output-dir    : Directory for output files
--verbose       : Enable detailed logging
--quiet         : Suppress non-essential output

For complete options, run: python audio_transcribe.py --help
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List, Set
from dataclasses import dataclass, field
import warnings
import tempfile
import shutil
import time
import subprocess
import glob
import json
from datetime import timedelta
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import hashlib

# Third-party imports
try:
    import numpy as np
    import whisper
    import soundfile as sf
    from scipy.io import wavfile
    from scipy.signal import butter, lfilter, sosfilt, zpk2sos
    from scipy import signal as scipy_signal
    import noisereduce as nr
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.utils import which
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("\nPlease install all required packages:")
    print("pip install openai-whisper pydub noisereduce numpy scipy soundfile tqdm")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global variables for cleanup
_temp_files: Set[Path] = set()
_cleanup_lock = threading.Lock()
_shutdown_requested = False

# Configure logging with more options
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )

# Default logging setup
setup_logging()
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("\nShutdown requested. Cleaning up...")
    cleanup_temp_files()
    sys.exit(0)


def cleanup_temp_files():
    """Clean up all temporary files."""
    global _temp_files
    with _cleanup_lock:
        for temp_file in _temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        _temp_files.clear()


def register_temp_file(file_path: Path) -> None:
    """Register a temporary file for cleanup."""
    global _temp_files
    with _cleanup_lock:
        _temp_files.add(file_path)


# Register signal handlers and cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_temp_files)


def check_disk_space(path: Path, required_mb: float = 500) -> bool:
    """Check if sufficient disk space is available."""
    try:
        stat = os.statvfs(path.parent if path.is_file() else path)
        available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        return available_mb >= required_mb
    except Exception:
        # If we can't check, assume it's okay
        return True


def validate_audio_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate an audio file is readable and not corrupted.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    if not file_path.is_file():
        return False, f"Not a file: {file_path}"

    if file_path.stat().st_size == 0:
        return False, f"File is empty: {file_path}"

    # Try to read file header
    try:
        # Try soundfile first
        with sf.SoundFile(str(file_path)) as f:
            if f.frames == 0:
                return False, f"Audio file has no frames: {file_path}"
            return True, None
    except Exception:
        # Try pydub as fallback
        try:
            audio = AudioSegment.from_file(str(file_path))
            if len(audio) == 0:
                return False, f"Audio file has zero duration: {file_path}"
            return True, None
        except Exception as e:
            return False, f"Cannot read audio file: {str(e)}"


def check_ffmpeg_availability() -> bool:
    """
    Check if FFmpeg is available in the system PATH.

    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    if which("ffmpeg") is None:
        logger.error("=" * 60)
        logger.error("ERROR: FFmpeg not found!")
        logger.error("=" * 60)
        logger.error("FFmpeg is required for audio processing.")
        logger.error("")
        logger.error("Installation instructions:")
        logger.error("  Windows : Use the PowerShell FFmpeg installer script")
        logger.error("  macOS   : brew install ffmpeg")
        logger.error("  Ubuntu  : sudo apt update && sudo apt install ffmpeg")
        logger.error("  Fedora  : sudo dnf install ffmpeg")
        logger.error("")
        logger.error("After installation, restart your terminal and try again.")
        logger.error("=" * 60)
        return False
    return True


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    elif seconds >= 10:
        return f"{secs}s"
    else:
        return f"{seconds:.1f}s"


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """
    Format seconds into timestamp string for subtitles.

    Args:
        seconds: Time in seconds
        fmt: Format type ('srt' or 'vtt')

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds = td.total_seconds() % 60

    if fmt == "srt":
        # SRT format: 00:00:00,000
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')
    else:
        # VTT format: 00:00:00.000
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def get_audio_info(file_path: Path) -> Dict[str, Any]:
    """Get detailed audio file information."""
    try:
        with sf.SoundFile(str(file_path)) as f:
            return {
                'duration': f.frames / f.samplerate,
                'sample_rate': f.samplerate,
                'channels': f.channels,
                'frames': f.frames,
                'format': f.format,
                'subtype': f.subtype,
                'size': file_path.stat().st_size
            }
    except Exception:
        # Fallback to pydub
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                'duration': len(audio) / 1000.0,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'frames': len(audio.get_array_of_samples()),
                'format': 'unknown',
                'subtype': 'unknown',
                'size': file_path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing parameters with gentler defaults."""
    # Core processing toggles
    normalize: bool = True
    compress_dynamics: bool = True
    reduce_noise: bool = True
    optimize_voice: bool = True

    # Gentle processing parameters
    noise_reduction_strength: float = 0.3  # Much gentler default (was 0.8)
    compression_threshold: float = -25.0   # Higher threshold (was -20)
    compression_ratio: float = 2.0         # Gentler ratio (was 4.0)

    # Subtle EQ adjustments (all reduced from original)
    voice_boost_db: float = 1.5            # Was 3.0
    clarity_boost_db: float = 1.0          # Was 2.0
    presence_boost_db: float = 0.5         # Was 1.0

    # Safety parameters
    target_level: float = 0.25             # Lower target (was 0.3)
    headroom_db: float = -3.0              # Leave headroom to prevent clipping
    limiter_threshold: float = 0.95        # Soft limiter threshold

    # Processing quality
    filter_order: int = 2                  # Gentler filters (was 5)
    processing_sample_rate: int = 16000    # Standard for speech

    # Performance options
    chunk_size: int = 1024 * 1024          # 1MB chunks for processing
    use_gpu: bool = False                  # GPU acceleration if available

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate noise reduction strength
        if not 0.0 <= self.noise_reduction_strength <= 1.0:
            raise ValueError(f"noise_reduction_strength must be between 0.0 and 1.0, "
                           f"got {self.noise_reduction_strength}")

        # Validate compression parameters
        if self.compression_threshold > 0:
            logger.warning(f"compression_threshold should typically be negative, "
                         f"got {self.compression_threshold}")

        if self.compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be >= 1.0, got {self.compression_ratio}")

        # Validate boost levels (warn if extreme)
        for name, value in [("voice_boost_db", self.voice_boost_db),
                           ("clarity_boost_db", self.clarity_boost_db),
                           ("presence_boost_db", self.presence_boost_db)]:
            if abs(value) > 6.0:
                logger.warning(f"{name} is high ({value}dB), may cause distortion")

        # Validate target level
        if not 0.0 < self.target_level <= 1.0:
            raise ValueError(f"target_level must be between 0.0 and 1.0, "
                           f"got {self.target_level}")

        # Validate headroom
        if self.headroom_db > 0:
            logger.warning("headroom_db should be negative")

        # Validate limiter threshold
        if not 0.5 <= self.limiter_threshold <= 1.0:
            raise ValueError(f"limiter_threshold must be between 0.5 and 1.0, "
                           f"got {self.limiter_threshold}")


@dataclass
class TranscriptionConfig:
    """Configuration for transcription parameters."""
    model_size: str = "base"
    language: Optional[str] = None
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None
    word_timestamps: bool = False
    verbose: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["txt"])

    # Additional options
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    patience: Optional[float] = None
    length_penalty: Optional[float] = None
    suppress_tokens: Optional[str] = None

    def __post_init__(self):
        """Validate transcription parameters."""
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if self.model_size not in valid_models:
            raise ValueError(f"Invalid model size: {self.model_size}. "
                           f"Valid options: {', '.join(valid_models)}")

        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")

        valid_formats = {'txt', 'json', 'srt', 'vtt'}
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid output format: {fmt}. "
                               f"Valid formats: {', '.join(valid_formats)}")


class OutputFormatter:
    """Handles formatting transcription results in various formats."""

    @staticmethod
    def save_txt(result: Dict[str, Any], output_path: Path) -> None:
        """Save transcription as plain text."""
        text = result.get("text", "").strip()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

            # Add metadata as comments
            f.write("\n\n" + "="*50 + "\n")
            f.write("# Transcription Metadata\n")
            f.write(f"# Language: {result.get('language', 'Unknown')}\n")
            f.write(f"# Duration: {result.get('duration', 0):.2f} seconds\n")
            f.write(f"# Segments: {len(result.get('segments', []))}\n")

            # Add processing timestamp
            from datetime import datetime
            f.write(f"# Processed: {datetime.now().isoformat()}\n")

    @staticmethod
    def save_json(result: Dict[str, Any], output_path: Path) -> None:
        """Save transcription with full metadata as JSON."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a clean output structure
        output = {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
            "segments": []
        }

        # Add segment information
        for segment in result.get("segments", []):
            seg_data = {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip()
            }

            # Add confidence scores if available
            if "avg_logprob" in segment:
                seg_data["confidence"] = np.exp(segment["avg_logprob"])

            # Add word-level timestamps if available
            if "words" in segment:
                seg_data["words"] = [
                    {
                        "word": word.get("word"),
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "probability": word.get("probability", 0)
                    }
                    for word in segment["words"]
                ]

            output["segments"].append(seg_data)

        # Add processing metadata
        from datetime import datetime
        output["metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "whisper_model": result.get("model", "unknown")
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_srt(result: Dict[str, Any], output_path: Path) -> None:
        """Save transcription as SRT subtitle file."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result.get("segments", []), 1):
                # Subtitle number
                f.write(f"{i}\n")

                # Timestamps
                start_time = format_timestamp(segment["start"], "srt")
                end_time = format_timestamp(segment["end"], "srt")
                f.write(f"{start_time} --> {end_time}\n")

                # Text (ensure it's not too long for subtitles)
                text = segment.get("text", "").strip()
                # Split long lines
                if len(text) > 80:
                    words = text.split()
                    lines = []
                    current_line = []
                    current_length = 0

                    for word in words:
                        if current_length + len(word) + 1 > 40:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += len(word) + 1

                    if current_line:
                        lines.append(' '.join(current_line))

                    text = '\n'.join(lines[:2])  # Max 2 lines

                f.write(f"{text}\n\n")

    @staticmethod
    def save_vtt(result: Dict[str, Any], output_path: Path) -> None:
        """Save transcription as WebVTT subtitle file."""
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            # WebVTT header
            f.write("WEBVTT\n")
            f.write("Kind: captions\n")
            f.write(f"Language: {result.get('language', 'en')}\n\n")

            for segment in result.get("segments", []):
                # Timestamps
                start_time = format_timestamp(segment["start"], "vtt")
                end_time = format_timestamp(segment["end"], "vtt")
                f.write(f"{start_time} --> {end_time}\n")

                # Text (with same line splitting as SRT)
                text = segment.get("text", "").strip()
                if len(text) > 80:
                    words = text.split()
                    lines = []
                    current_line = []
                    current_length = 0

                    for word in words:
                        if current_length + len(word) + 1 > 40:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += len(word) + 1

                    if current_line:
                        lines.append(' '.join(current_line))

                    text = '\n'.join(lines[:2])

                f.write(f"{text}\n\n")


class AudioProcessor:
    """
    Audio processing class for improving voice clarity with gentle, stable processing.

    This version uses conservative processing to avoid distortion while
    still improving speech intelligibility.
    """

    def __init__(self, audio_path: Union[str, Path], config: Optional[AudioProcessingConfig] = None):
        """
        Initialize the AudioProcessor.

        Args:
            audio_path: Path to the input audio file
            config: Processing configuration (uses defaults if None)
        """
        self.audio_path = Path(audio_path)
        self.config = config or AudioProcessingConfig()
        self._validate_input()
        self._processed_hash = None

    def _validate_input(self) -> None:
        """Validate that the input file exists and is readable."""
        is_valid, error_msg = validate_audio_file(self.audio_path)
        if not is_valid:
            raise ValueError(error_msg)

    def get_file_hash(self) -> str:
        """Get hash of the audio file for caching purposes."""
        if self._processed_hash is None:
            hasher = hashlib.md5()
            with open(self.audio_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            self._processed_hash = hasher.hexdigest()
        return self._processed_hash

    def process(self, output_path: Optional[Union[str, Path]] = None,
                progress_callback: Optional[callable] = None) -> str:
        """
        Apply gentle audio processing pipeline optimized for speech.

        Args:
            output_path: Optional path for processed audio output
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the processed audio file

        Raises:
            RuntimeError: If processing fails
        """
        if output_path is None:
            output_path = self.audio_path.parent / f"{self.audio_path.stem}_processed.wav"
        else:
            output_path = Path(output_path)

        # Check disk space
        file_size = self.audio_path.stat().st_size
        required_space = file_size * 3  # Conservative estimate
        if not check_disk_space(output_path, required_space / (1024 * 1024)):
            raise RuntimeError(f"Insufficient disk space. Need at least {format_size(required_space)}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if shutdown was requested
        if _shutdown_requested:
            raise RuntimeError("Processing cancelled by user")

        try:
            logger.info("Starting gentle audio processing pipeline...")
            logger.info("-" * 50)

            # Load audio
            logger.info("Loading audio file...")
            if progress_callback:
                progress_callback(0.1, "Loading audio...")

            samples, original_sr = self._load_audio_safely()

            # Resample to standard speech processing rate if needed
            if original_sr != self.config.processing_sample_rate:
                logger.info(f"Resampling from {original_sr}Hz to {self.config.processing_sample_rate}Hz...")
                if progress_callback:
                    progress_callback(0.2, "Resampling audio...")

                samples = self._resample_audio(samples, original_sr, self.config.processing_sample_rate)
                sample_rate = self.config.processing_sample_rate
            else:
                sample_rate = original_sr

            # Apply processing stages in optimal order
            stage_progress = 0.2
            stage_increment = 0.6 / 4  # 4 main stages

            if self.config.reduce_noise:
                logger.info("Applying gentle noise reduction...")
                if progress_callback:
                    progress_callback(stage_progress, "Reducing noise...")
                samples = self._gentle_noise_reduction(samples, sample_rate)
                stage_progress += stage_increment

            if self.config.optimize_voice:
                logger.info("Optimizing voice frequencies...")
                if progress_callback:
                    progress_callback(stage_progress, "Optimizing voice...")
                samples = self._optimize_voice_eq(samples, sample_rate)
                stage_progress += stage_increment

            if self.config.compress_dynamics:
                logger.info("Applying soft compression...")
                if progress_callback:
                    progress_callback(stage_progress, "Compressing dynamics...")
                samples = self._soft_compress(samples, sample_rate)
                stage_progress += stage_increment

            if self.config.normalize:
                logger.info("Normalizing levels with headroom...")
                if progress_callback:
                    progress_callback(stage_progress, "Normalizing...")
                samples = self._normalize_with_headroom(samples)

            # Final safety limiter
            logger.info("Applying final limiter...")
            if progress_callback:
                progress_callback(0.9, "Finalizing...")
            samples = self._soft_limit(samples)

            # Save processed audio
            logger.info("Saving processed audio...")
            if progress_callback:
                progress_callback(0.95, "Saving...")

            sf.write(str(output_path), samples, sample_rate, subtype='PCM_16')

            if progress_callback:
                progress_callback(1.0, "Complete")

            logger.info("-" * 50)
            logger.info(f"✓ Processed audio saved to: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Audio processing failed: {str(e)}")

    def _load_audio_safely(self) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to mono float32 samples."""
        try:
            # Try soundfile first (more reliable)
            samples, sample_rate = sf.read(str(self.audio_path), dtype='float32', always_2d=False)

            # Convert to mono if stereo
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)

            # Validate samples
            if not np.isfinite(samples).all():
                raise ValueError("Audio contains invalid samples (inf or nan)")

            logger.info(f"Audio loaded: {len(samples)/sample_rate:.2f}s @ {sample_rate}Hz")
            return samples, sample_rate

        except Exception as e:
            logger.warning(f"Soundfile failed, trying pydub: {e}")

            # Fallback to pydub
            try:
                audio = AudioSegment.from_file(str(self.audio_path))
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

                # Normalize to [-1, 1]
                max_val = float(2 ** (audio.sample_width * 8 - 1))
                samples = samples / max_val

                # Convert stereo to mono
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)

                # Validate
                if not np.isfinite(samples).all():
                    raise ValueError("Audio contains invalid samples")

                return samples, audio.frame_rate
            except Exception as e:
                raise RuntimeError(f"Failed to load audio: {str(e)}")

    def _resample_audio(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate with high quality."""
        try:
            # Use high-quality resampling
            num_samples = int(len(samples) * target_sr / orig_sr)

            # Use Fourier method for best quality
            resampled = scipy_signal.resample(samples, num_samples, window='hamming')

            # Remove any DC offset introduced by resampling
            resampled = resampled - np.mean(resampled)

            return resampled
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise RuntimeError(f"Failed to resample audio: {str(e)}")

    def _gentle_noise_reduction(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle noise reduction that preserves speech."""
        try:
            # Estimate noise profile from quiet parts
            # Find quietest 10% of the signal for noise profile
            frame_length = int(sample_rate * 0.1)  # 100ms frames
            energies = []

            for i in range(0, len(samples) - frame_length, frame_length // 2):
                frame = samples[i:i + frame_length]
                energy = np.sqrt(np.mean(frame ** 2))
                energies.append((energy, i))

            # Sort by energy and take lowest 10%
            energies.sort(key=lambda x: x[0])
            noise_frames = energies[:max(1, len(energies) // 10)]

            # Use stationary noise reduction with gentle settings
            reduced = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                prop_decrease=self.config.noise_reduction_strength,
                stationary=True,  # Better for consistent background noise
                use_tqdm=False
            )

            # Mix with original to preserve speech naturalness
            # This prevents over-processing
            mix_ratio = 0.7  # 70% processed, 30% original
            output = reduced * mix_ratio + samples * (1 - mix_ratio)

            # Remove any DC offset
            output = output - np.mean(output)

            return output

        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Using original audio.")
            return samples

    def _optimize_voice_eq(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle EQ optimized for voice clarity."""
        try:
            # Use second-order sections for stable filtering
            nyquist = sample_rate * 0.5

            # Copy samples to avoid modifying original
            output = samples.copy()

            # High-pass to remove rumble (80Hz)
            if 80 < nyquist:
                sos_hp = self._design_highpass(80, sample_rate)
                output = sosfilt(sos_hp, output)

            # Gentle voice presence boost (200-400Hz)
            if 400 < nyquist and self.config.voice_boost_db > 0:
                output = self._apply_bell_eq(
                    output, 300, 1.5, self.config.voice_boost_db, sample_rate
                )

            # Clarity boost (2-4kHz) - important for consonants
            if 4000 < nyquist and self.config.clarity_boost_db > 0:
                output = self._apply_bell_eq(
                    output, 3000, 1.0, self.config.clarity_boost_db, sample_rate
                )

            # Very gentle high shelf for "air" (8kHz+)
            if 8000 < nyquist and self.config.presence_boost_db > 0:
                output = self._apply_shelf_eq(
                    output, 8000, self.config.presence_boost_db, sample_rate, 'high'
                )

            return output
        except Exception as e:
            logger.warning(f"EQ optimization failed: {e}. Using original audio.")
            return samples

    def _design_highpass(self, freq: float, sample_rate: int) -> np.ndarray:
        """Design stable highpass filter using zpk method."""
        nyquist = sample_rate * 0.5
        normalized_freq = min(freq / nyquist, 0.99)  # Ensure < 1

        # Use zpk design for better numerical stability
        z, p, k = butter(self.config.filter_order, normalized_freq,
                        btype='high', analog=False, output='zpk')

        # Convert to second-order sections
        sos = zpk2sos(z, p, k)
        return sos

    def _apply_bell_eq(self, samples: np.ndarray, center_freq: float,
                      q_factor: float, gain_db: float, sample_rate: int) -> np.ndarray:
        """Apply bell-shaped EQ boost/cut at specific frequency."""
        try:
            # Limit gain to prevent instability
            gain_db = np.clip(gain_db, -12, 12)

            # Simple peak EQ using biquad coefficients
            w0 = 2 * np.pi * center_freq / sample_rate
            cos_w0 = np.cos(w0)
            sin_w0 = np.sin(w0)

            A = 10 ** (gain_db / 40)
            alpha = sin_w0 / (2 * q_factor)

            # Peaking EQ coefficients
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

            # Normalize
            b = np.array([b0, b1, b2]) / a0
            a = np.array([1, a1/a0, a2/a0])

            # Apply filter with initial conditions to avoid clicks
            zi = lfilter([1], a, samples[:2])
            filtered, _ = lfilter(b, a, samples, zi=zi*b[0])

            return filtered
        except Exception as e:
            logger.warning(f"Bell EQ failed: {e}")
            return samples

    def _apply_shelf_eq(self, samples: np.ndarray, freq: float,
                       gain_db: float, sample_rate: int, shelf_type: str) -> np.ndarray:
        """Apply gentle shelf EQ."""
        try:
            nyquist = sample_rate * 0.5
            normalized_freq = min(freq / nyquist, 0.95)

            # Design shelf filter
            if shelf_type == 'high':
                z, p, k = butter(2, normalized_freq, btype='high', analog=False, output='zpk')
            else:
                z, p, k = butter(2, normalized_freq, btype='low', analog=False, output='zpk')

            sos = zpk2sos(z, p, k)
            filtered = sosfilt(sos, samples)

            # Apply gain
            gain_linear = 10 ** (gain_db / 20)
            boosted = samples + filtered * (gain_linear - 1)

            # Gentle limiting to prevent clipping
            return np.tanh(boosted * 0.9) / 0.9
        except Exception as e:
            logger.warning(f"Shelf EQ failed: {e}")
            return samples

    def _soft_compress(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply soft knee compression for consistent levels."""
        try:
            # Simple RMS-based compressor with soft knee
            frame_length = int(sample_rate * 0.010)  # 10ms frames
            hop_length = frame_length // 2

            threshold_linear = 10 ** (self.config.compression_threshold / 20)
            ratio = self.config.compression_ratio

            # Soft knee width (in dB)
            knee_width = 6.0
            knee_start = threshold_linear * 10 ** (-knee_width / 40)
            knee_end = threshold_linear * 10 ** (knee_width / 40)

            output = np.zeros_like(samples)

            # Calculate envelope with RMS
            envelope = np.zeros(len(samples))
            for i in range(0, len(samples) - frame_length, hop_length):
                frame = samples[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2) + 1e-10)  # Add small value to avoid log(0)
                envelope[i:i + hop_length] = rms

            # Fill the end
            envelope[len(envelope) - hop_length:] = envelope[len(envelope) - hop_length - 1]

            # Smooth envelope
            from scipy.ndimage import gaussian_filter1d
            envelope = gaussian_filter1d(envelope, sigma=sample_rate * 0.005)  # 5ms smoothing

            # Apply compression
            gain = np.ones_like(envelope)

            # Below knee - no compression
            mask_below = envelope < knee_start
            gain[mask_below] = 1.0

            # In knee region - smooth transition
            mask_knee = (envelope >= knee_start) & (envelope <= knee_end)
            if np.any(mask_knee):
                knee_factor = (envelope[mask_knee] - knee_start) / (knee_end - knee_start)
                knee_factor = 0.5 * (1 - np.cos(np.pi * knee_factor))

                full_gain_reduction = (envelope[mask_knee] / threshold_linear) ** (1 - 1/ratio)
                full_gain = threshold_linear / envelope[mask_knee] * full_gain_reduction

                gain[mask_knee] = 1.0 * (1 - knee_factor) + full_gain * knee_factor

            # Above knee - full compression
            mask_above = envelope > knee_end
            if np.any(mask_above):
                gain_reduction = (envelope[mask_above] / threshold_linear) ** (1 - 1/ratio)
                gain[mask_above] = threshold_linear / envelope[mask_above] * gain_reduction

            # Apply gain with attack/release
            attack_time = 0.005  # 5ms
            release_time = 0.050  # 50ms

            attack_samples = int(attack_time * sample_rate)
            release_samples = int(release_time * sample_rate)

            smoothed_gain = np.zeros_like(gain)
            smoothed_gain[0] = gain[0]

            for i in range(1, len(gain)):
                if gain[i] < smoothed_gain[i-1]:
                    # Attack
                    alpha = 1.0 - np.exp(-1.0 / attack_samples)
                else:
                    # Release
                    alpha = 1.0 - np.exp(-1.0 / release_samples)

                smoothed_gain[i] = smoothed_gain[i-1] + alpha * (gain[i] - smoothed_gain[i-1])

            # Apply smoothed gain
            output = samples * smoothed_gain

            return output
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return samples

    def _normalize_with_headroom(self, samples: np.ndarray) -> np.ndarray:
        """Normalize audio with headroom to prevent clipping."""
        try:
            # Find peak using percentile to ignore outliers
            peak = np.percentile(np.abs(samples), 99.9)

            if peak > 0:
                # Calculate gain to reach target with headroom
                target_peak = 10 ** (self.config.headroom_db / 20)  # Convert dB to linear
                gain = target_peak / peak

                # Limit gain to prevent over-amplification
                max_gain = 10.0  # 20dB maximum gain
                gain = min(gain, max_gain)

                # Apply gain with soft limiting
                normalized = samples * gain

                # Soft limit any remaining peaks
                return self._soft_limit(normalized, threshold=target_peak)

            return samples
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return samples

    def _soft_limit(self, samples: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Apply soft limiting to prevent clipping while maintaining quality."""
        try:
            if threshold is None:
                threshold = self.config.limiter_threshold

            # Soft clipping using tanh (hyperbolic tangent)
            # This provides smooth limiting without hard clipping

            # Scale samples so threshold maps to tanh inflection point
            scale_factor = 1.0 / threshold
            scaled = samples * scale_factor

            # Apply soft limiting
            limited = np.tanh(scaled * 0.7) / 0.7  # 0.7 makes the knee softer

            # Scale back
            output = limited / scale_factor

            # Ensure we don't exceed [-1, 1]
            output = np.clip(output, -1.0, 1.0)

            return output
        except Exception as e:
            logger.warning(f"Limiting failed: {e}")
            return np.clip(samples, -1.0, 1.0)


class AudioTranscriber:
    """
    Main transcription class using OpenAI's Whisper.

    Handles the transcription process with optional audio preprocessing
    and various output formats.
    """

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Initialize the AudioTranscriber.

        Args:
            config: Transcription configuration (uses defaults if None)
        """
        self.config = config or TranscriptionConfig()
        self.model = None
        self.formatter = OutputFormatter()
        self._model_lock = threading.Lock()

    def _cleanup_on_exit(self):
        """Cleanup when transcriber is destroyed."""
        # Model cleanup is handled by whisper
        pass

    def load_model(self) -> None:
        """Load the Whisper model."""
        with self._model_lock:
            if self.model is None:
                logger.info(f"Loading Whisper {self.config.model_size} model...")
                logger.info("This may take a moment for first-time use...")

                try:
                    # Check if model is already downloaded
                    import whisper
                    model_path = whisper._download(whisper._MODELS[self.config.model_size],
                                                  root=os.path.expanduser("~/.cache/whisper"),
                                                  in_memory=False)

                    # Load model
                    self.model = whisper.load_model(self.config.model_size)
                    logger.info(f"✓ Model loaded successfully: {self.config.model_size}")

                except Exception as e:
                    # Try to provide more helpful error messages
                    if "CUDA" in str(e):
                        logger.error("CUDA error detected. Falling back to CPU...")
                        os.environ["CUDA_VISIBLE_DEVICES"] = ""
                        try:
                            self.model = whisper.load_model(self.config.model_size, device="cpu")
                            logger.info(f"✓ Model loaded on CPU: {self.config.model_size}")
                        except Exception as cpu_error:
                            raise RuntimeError(f"Failed to load model on CPU: {str(cpu_error)}")
                    else:
                        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    def transcribe(self, audio_path: Union[str, Path],
                  output_path: Optional[Union[str, Path]] = None,
                  preprocess: bool = True,
                  processing_config: Optional[AudioProcessingConfig] = None,
                  save_processed: bool = True,
                  progress_callback: Optional[callable] = None) -> Tuple[str, str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to input audio file
            output_path: Optional base path for output files (without extension)
            preprocess: Whether to preprocess audio before transcription
            processing_config: Optional audio processing configuration
            save_processed: Whether to keep the processed audio file
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (transcribed text, primary output file path)

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If transcription fails
        """
        audio_path = Path(audio_path)

        # Validate input file
        is_valid, error_msg = validate_audio_file(audio_path)
        if not is_valid:
            raise FileNotFoundError(error_msg)

        # Get audio info for logging
        audio_info = get_audio_info(audio_path)
        logger.info(f"Input audio: {format_time(audio_info.get('duration', 0))}, "
                   f"{audio_info.get('sample_rate', 0)}Hz, "
                   f"{format_size(audio_info.get('size', 0))}")

        # Determine output base path (without extension)
        if output_path is None:
            output_base = audio_path.parent / f"{audio_path.stem}_transcript"
        else:
            output_base = Path(str(output_path).rsplit('.', 1)[0])  # Remove extension if present

        # Ensure output directory exists
        output_base.parent.mkdir(parents=True, exist_ok=True)

        # Load model if not already loaded
        self.load_model()

        # Process audio
        transcription_audio_path = str(audio_path)
        processed_audio_path = None

        try:
            # Update progress
            if progress_callback:
                progress_callback(0.0, "Starting transcription process...")

            if preprocess:
                logger.info("Preprocessing audio for better transcription...")
                processor = AudioProcessor(str(audio_path), processing_config)

                # Create processed audio in temp location if not saving
                if save_processed:
                    processed_audio_path = audio_path.parent / f"{audio_path.stem}_processed.wav"
                else:
                    # Use temp file
                    temp_fd, processed_audio_path = tempfile.mkstemp(suffix='.wav',
                                                                     prefix=f"{audio_path.stem}_")
                    os.close(temp_fd)  # Close file descriptor
                    register_temp_file(Path(processed_audio_path))

                # Process with progress updates
                def processing_progress(progress, status):
                    if progress_callback:
                        # Processing is 0-40% of total progress
                        progress_callback(progress * 0.4, f"Processing: {status}")

                processed_audio_path = processor.process(processed_audio_path, processing_progress)
                transcription_audio_path = processed_audio_path

            # Transcribe
            logger.info("Starting transcription...")
            if progress_callback:
                progress_callback(0.4, "Transcribing audio...")

            result = self._transcribe_audio(transcription_audio_path, progress_callback)

            # Save transcripts in requested formats
            if progress_callback:
                progress_callback(0.9, "Saving transcripts...")

            output_files = self._save_transcripts(result, output_base)

            # Clean up temporary processed audio if not saving
            if preprocess and not save_processed and processed_audio_path:
                Path(processed_audio_path).unlink()
                logger.info("Temporary processed audio removed")

            # Log results
            self._log_results(result, output_files,
                            processed_audio_path if (preprocess and save_processed) else None)

            if progress_callback:
                progress_callback(1.0, "Complete!")

            # Return text and primary output file (first format)
            return result["text"].strip(), str(output_files[0])

        except Exception as e:
            # Clean up on error
            if processed_audio_path and not save_processed:
                try:
                    Path(processed_audio_path).unlink()
                except Exception:
                    pass
            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def _transcribe_audio(self, audio_path: str,
                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Execute the Whisper transcription with progress indication."""
        try:
            # Get audio duration for progress estimation
            audio_info = get_audio_info(Path(audio_path))
            duration_seconds = audio_info.get('duration', 0)

            logger.info(f"Transcribing {format_time(duration_seconds)} of audio...")

            start_time = time.time()

            # Prepare transcription options
            transcribe_options = {
                'language': self.config.language,
                'task': 'transcribe',
                'temperature': self.config.temperature,
                'compression_ratio_threshold': self.config.compression_ratio_threshold,
                'logprob_threshold': self.config.logprob_threshold,
                'no_speech_threshold': self.config.no_speech_threshold,
                'condition_on_previous_text': self.config.condition_on_previous_text,
                'initial_prompt': self.config.initial_prompt,
                'word_timestamps': self.config.word_timestamps,
                'prepend_punctuations': "\"'([{-",
                'append_punctuations': "\"'.。,，!！?？:：)]}、",
                'verbose': self.config.verbose
            }

            # Add optional parameters
            if self.config.beam_size is not None:
                transcribe_options['beam_size'] = self.config.beam_size
            if self.config.best_of is not None:
                transcribe_options['best_of'] = self.config.best_of
            if self.config.patience is not None:
                transcribe_options['patience'] = self.config.patience
            if self.config.length_penalty is not None:
                transcribe_options['length_penalty'] = self.config.length_penalty
            if self.config.suppress_tokens is not None:
                transcribe_options['suppress_tokens'] = self.config.suppress_tokens

            # Progress tracking for verbose mode
            if self.config.verbose and progress_callback:
                # Create a custom progress hook
                segments_processed = 0

                def progress_hook(segments):
                    nonlocal segments_processed
                    segments_processed = len(segments)
                    # Estimate progress (40-90% range for transcription)
                    progress = 0.4 + (segments_processed / max(1, duration_seconds / 30)) * 0.5
                    progress = min(0.9, progress)
                    if progress_callback:
                        progress_callback(progress, f"Processing segment {segments_processed}...")

                # Note: Whisper doesn't have built-in progress callbacks,
                # so we estimate based on processing time
                import threading

                def progress_monitor():
                    while not hasattr(progress_monitor, 'stop'):
                        elapsed = time.time() - start_time
                        # Rough estimate: 1 second of audio takes ~0.1s to process
                        estimated_progress = min(0.9, 0.4 + (elapsed / (duration_seconds * 0.1)) * 0.5)
                        if progress_callback:
                            progress_callback(estimated_progress, "Transcribing...")
                        time.sleep(1)

                monitor_thread = threading.Thread(target=progress_monitor)
                monitor_thread.daemon = True
                monitor_thread.start()

            # Transcribe
            result = self.model.transcribe(audio_path, **transcribe_options)

            # Stop progress monitor if running
            if 'monitor_thread' in locals():
                progress_monitor.stop = True
                monitor_thread.join(timeout=1)

            elapsed_time = time.time() - start_time
            logger.info(f"Transcription completed in {format_time(elapsed_time)}")

            # Add metadata
            result['duration'] = duration_seconds
            result['model'] = self.config.model_size

            return result
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError("Out of memory. Try using a smaller model or shorter audio segments.")
            else:
                raise RuntimeError(f"Whisper transcription failed: {str(e)}")

    def _save_transcripts(self, result: Dict[str, Any], output_base: Path) -> List[Path]:
        """
        Save transcription results in requested formats.

        Args:
            result: Transcription result from Whisper
            output_base: Base path for output files (without extension)

        Returns:
            List of output file paths
        """
        output_files = []

        for fmt in self.config.output_formats:
            try:
                if fmt == "txt":
                    output_path = output_base.with_suffix(".txt")
                    self.formatter.save_txt(result, output_path)
                elif fmt == "json":
                    output_path = output_base.with_suffix(".json")
                    self.formatter.save_json(result, output_path)
                elif fmt == "srt":
                    output_path = output_base.with_suffix(".srt")
                    self.formatter.save_srt(result, output_path)
                elif fmt == "vtt":
                    output_path = output_base.with_suffix(".vtt")
                    self.formatter.save_vtt(result, output_path)
                else:
                    logger.warning(f"Unknown output format: {fmt}")
                    continue

                output_files.append(output_path)
                logger.debug(f"Saved {fmt.upper()} format to: {output_path}")

            except Exception as e:
                logger.error(f"Failed to save {fmt} format: {e}")

        if not output_files:
            raise RuntimeError("Failed to save any output files")

        return output_files

    def _log_results(self, result: Dict[str, Any], output_files: List[Path],
                    processed_path: Optional[str]) -> None:
        """Log transcription results."""
        logger.info("=" * 50)
        logger.info("Transcription completed successfully!")

        for output_file in output_files:
            logger.info(f"Output saved to: {output_file}")

        if processed_path:
            logger.info(f"Processed audio saved to: {processed_path}")

        logger.info(f"Detected language: {result.get('language', 'Unknown')}")
        logger.info(f"Duration: {result.get('duration', 0):.2f} seconds")
        logger.info(f"Number of segments: {len(result.get('segments', []))}")

        # Calculate average confidence if available
        segments = result.get('segments', [])
        if segments and 'avg_logprob' in segments[0]:
            avg_confidence = np.mean([np.exp(seg.get('avg_logprob', 0)) for seg in segments])
            logger.info(f"Average confidence: {avg_confidence:.2%}")

        # Preview
        text = result.get("text", "").strip()
        if text:
            preview_length = 500
            if len(text) > preview_length:
                preview = text[:preview_length] + "..."
            else:
                preview = text

            logger.info("-" * 50)
            logger.info("Transcript preview:")
            logger.info(preview)
        else:
            logger.warning("No text was transcribed")

        logger.info("=" * 50)


def collect_audio_files(args: argparse.Namespace) -> List[Path]:
    """
    Collect all audio files to process based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of Path objects for audio files to process
    """
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.wma',
                       '.aac', '.opus', '.webm', '.m4b', '.mp4', '.avi', '.mkv'}
    files = []

    if args.input_dir:
        # Process directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if not input_dir.is_dir():
            raise ValueError(f"Not a directory: {input_dir}")

        pattern = args.file_pattern
        logger.info(f"Searching for audio files in: {input_dir}")

        if args.recursive:
            # Recursive search
            for ext in audio_extensions:
                files.extend(input_dir.rglob(f"{pattern}{ext}"))
                files.extend(input_dir.rglob(f"{pattern}{ext.upper()}"))
        else:
            # Non-recursive search
            for ext in audio_extensions:
                files.extend(input_dir.glob(f"{pattern}{ext}"))
                files.extend(input_dir.glob(f"{pattern}{ext.upper()}"))

        logger.info(f"Found {len(files)} audio files in directory")
    else:
        # Process provided files/patterns
        for file_pattern in args.audio_files:
            # Check if it's a glob pattern
            if '*' in file_pattern or '?' in file_pattern:
                matched_files = glob.glob(file_pattern, recursive=True)
                for f in matched_files:
                    path = Path(f)
                    if path.suffix.lower() in audio_extensions:
                        files.append(path)
                    else:
                        logger.debug(f"Skipping non-audio file: {path}")
            else:
                # Single file
                path = Path(file_pattern)
                if path.exists() and path.suffix.lower() in audio_extensions:
                    files.append(path)
                elif path.exists():
                    logger.warning(f"Skipping non-audio file: {path}")
                else:
                    raise FileNotFoundError(f"File not found: {path}")

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(f)

    # Validate all files
    valid_files = []
    for f in unique_files:
        is_valid, error_msg = validate_audio_file(f)
        if is_valid:
            valid_files.append(f)
        else:
            logger.warning(f"Skipping invalid file: {error_msg}")

    return valid_files


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if input is provided
    if not args.audio_files and not args.input_dir:
        raise ValueError("No input files specified. Provide audio files or use --input-dir")

    # Check for conflicting options
    if args.audio_files and args.input_dir:
        raise ValueError("Cannot specify both audio files and --input-dir. Choose one input method")

    # Validate single file output restriction
    if args.output and args.input_dir:
        raise ValueError("Cannot use -o/--output with --input-dir. Use --output-dir instead")

    if args.output and args.audio_files and len(args.audio_files) > 1:
        raise ValueError("Cannot use -o/--output with multiple input files. Use --output-dir instead")

    # Validate noise reduction strength
    if not 0.0 <= args.noise_reduction <= 1.0:
        raise ValueError("Noise reduction strength must be between 0.0 and 1.0")

    # Validate temperature
    if args.temperature < 0.0:
        raise ValueError("Temperature must be non-negative")

    # Validate output formats
    valid_formats = {'txt', 'json', 'srt', 'vtt'}
    for fmt in args.format:
        if fmt not in valid_formats:
            raise ValueError(f"Invalid output format: {fmt}. Valid formats: {', '.join(valid_formats)}")

    # Validate thread count
    if args.threads < 1:
        raise ValueError("Thread count must be at least 1")

    # Validate log level
    valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    if args.log_level.upper() not in valid_log_levels:
        raise ValueError(f"Invalid log level: {args.log_level}. "
                        f"Valid levels: {', '.join(valid_log_levels)}")


class BatchProcessor:
    """Handles batch processing of multiple audio files with progress tracking."""

    def __init__(self, transcriber: AudioTranscriber, args: argparse.Namespace):
        self.transcriber = transcriber
        self.args = args
        self.results = []
        self.start_time = None
        self.lock = threading.Lock()

    def process_files(self, files: List[Path]) -> Dict[str, Any]:
        """
        Process multiple audio files and return summary statistics.

        Args:
            files: List of audio files to process

        Returns:
            Dictionary with processing summary
        """
        self.start_time = time.time()
        total_files = len(files)
        successful = 0
        failed = 0
        self.results = []

        logger.info("=" * 60)
        logger.info(f"Batch Processing: {total_files} file(s)")
        if self.args.threads > 1:
            logger.info(f"Using {self.args.threads} parallel threads")
        logger.info("=" * 60)

        if self.args.threads > 1 and total_files > 1:
            # Parallel processing
            successful, failed = self._process_parallel(files)
        else:
            # Sequential processing
            successful, failed = self._process_sequential(files)

        # Summary
        total_time = time.time() - self.start_time
        return {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'total_time': total_time,
            'results': self.results
        }

    def _process_sequential(self, files: List[Path]) -> Tuple[int, int]:
        """Process files sequentially."""
        successful = 0
        failed = 0
        total_files = len(files)

        for idx, audio_file in enumerate(files, 1):
            if _shutdown_requested:
                logger.info("Batch processing interrupted by user")
                break

            file_start_time = time.time()

            # Progress header
            logger.info("")
            logger.info(f"[{idx}/{total_files}] Processing: {audio_file.name}")
            logger.info("-" * 50)

            try:
                # Process file
                result = self._process_single_file(audio_file, idx, total_files)

                with self.lock:
                    self.results.append(result)

                if result['success']:
                    successful += 1
                    if not self.args.quiet:
                        logger.info(f"✓ Completed in {format_time(result['time'])}")
                else:
                    failed += 1

            except Exception as e:
                # Track failure
                failed += 1
                file_time = time.time() - file_start_time

                with self.lock:
                    self.results.append({
                        'file': audio_file,
                        'output': None,
                        'success': False,
                        'time': file_time,
                        'error': str(e)
                    })

                logger.error(f"✗ Failed: {str(e)}")
                if self.args.verbose:
                    logger.exception("Detailed error:")

        return successful, failed

    def _process_parallel(self, files: List[Path]) -> Tuple[int, int]:
        """Process files in parallel using thread pool."""
        successful = 0
        failed = 0
        total_files = len(files)

        # Progress bar for parallel processing
        pbar = None
        if not self.args.quiet:
            try:
                pbar = tqdm(total=total_files, desc="Processing files", unit="files")
            except Exception:
                # Fallback if tqdm fails
                pass

        with ThreadPoolExecutor(max_workers=self.args.threads) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_file, audio_file, idx, total_files): (audio_file, idx)
                for idx, audio_file in enumerate(files, 1)
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                if _shutdown_requested:
                    executor.shutdown(wait=False)
                    break

                audio_file, idx = future_to_file[future]

                try:
                    result = future.result()

                    with self.lock:
                        self.results.append(result)

                    if result['success']:
                        successful += 1
                    else:
                        failed += 1

                    if pbar:
                        pbar.update(1)

                except Exception as e:
                    # Track failure
                    failed += 1

                    with self.lock:
                        self.results.append({
                            'file': audio_file,
                            'output': None,
                            'success': False,
                            'time': 0,
                            'error': str(e)
                        })

                    logger.error(f"✗ [{idx}/{total_files}] {audio_file.name}: {str(e)}")

                    if pbar:
                        pbar.update(1)

        if pbar:
            pbar.close()

        return successful, failed

    def _process_single_file(self, audio_file: Path, idx: int, total: int) -> Dict[str, Any]:
        """Process a single audio file."""
        file_start_time = time.time()

        try:
            # Determine output path
            output_path = self._get_output_path(audio_file)

            # Get file info
            file_size = audio_file.stat().st_size / (1024 * 1024)  # MB

            if not self.args.quiet and self.args.threads == 1:
                logger.info(f"File size: {file_size:.1f} MB")

            # Progress callback for single-threaded mode
            progress_callback = None
            if self.args.threads == 1 and not self.args.quiet:
                def progress_callback(progress, status):
                    # Simple progress indication
                    pass

            # Process file
            text, output_file = self.transcriber.transcribe(
                audio_path=audio_file,
                output_path=output_path,
                preprocess=not self.args.no_preprocess,
                processing_config=self._get_processing_config(),
                save_processed=self.args.save_processed,
                progress_callback=progress_callback
            )

            # Track success
            file_time = time.time() - file_start_time

            return {
                'file': audio_file,
                'output': output_file,
                'success': True,
                'time': file_time,
                'error': None,
                'text_preview': text[:100] + "..." if len(text) > 100 else text
            }

        except Exception as e:
            # Track failure
            file_time = time.time() - file_start_time

            return {
                'file': audio_file,
                'output': None,
                'success': False,
                'time': file_time,
                'error': str(e),
                'text_preview': None
            }

    def _get_output_path(self, audio_file: Path) -> Optional[Path]:
        """Determine output path for a given audio file."""
        if self.args.output_dir:
            output_dir = Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Preserve relative directory structure if processing directories
            if self.args.input_dir:
                try:
                    rel_path = audio_file.relative_to(Path(self.args.input_dir))
                    output_path = output_dir / rel_path.parent / audio_file.stem
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except ValueError:
                    # If file is not relative to input_dir, just use filename
                    output_path = output_dir / audio_file.stem
            else:
                output_path = output_dir / audio_file.stem

            return output_path
        elif self.args.output:
            # Single file with specified output
            return Path(self.args.output)
        else:
            # Default: same directory as input
            return None

    def _get_processing_config(self) -> Optional[AudioProcessingConfig]:
        """Get audio processing configuration from command line args."""
        if not self.args.no_preprocess:
            return AudioProcessingConfig(
                noise_reduction_strength=self.args.noise_reduction,
                voice_boost_db=self.args.voice_boost,
                clarity_boost_db=self.args.clarity_boost,
                presence_boost_db=self.args.presence_boost
            )
        return None

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print batch processing summary."""
        if self.args.quiet:
            return

        print("\n" + "=" * 60)
        print("Batch Processing Summary")
        print("=" * 60)
        print(f"Total files    : {summary['total_files']}")
        print(f"Successful     : {summary['successful']} ✓")
        print(f"Failed         : {summary['failed']} ✗")
        print(f"Total time     : {format_time(summary['total_time'])}")

        if summary['total_files'] > 0:
            avg_time = summary['total_time'] / summary['total_files']
            print(f"Average time   : {format_time(avg_time)} per file")

            # Calculate throughput
            if summary['total_time'] > 0:
                throughput = summary['total_files'] / (summary['total_time'] / 60)
                print(f"Throughput     : {throughput:.1f} files/minute")

        if summary['failed'] > 0:
            print("\nFailed files:")
            for result in summary['results']:
                if not result['success']:
                    print(f"  ✗ {result['file'].name}: {result['error']}")

        # Save summary to file if requested
        if self.args.summary_file:
            summary_path = Path(self.args.summary_file)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': {
                        'total_files': summary['total_files'],
                        'successful': summary['successful'],
                        'failed': summary['failed'],
                        'total_time': summary['total_time'],
                        'average_time': summary['total_time'] / max(1, summary['total_files'])
                    },
                    'results': [
                        {
                            'file': str(r['file']),
                            'output': str(r['output']) if r['output'] else None,
                            'success': r['success'],
                            'time': r['time'],
                            'error': r['error'],
                            'preview': r.get('text_preview', None)
                        }
                        for r in summary['results']
                    ]
                }, f, indent=2, ensure_ascii=False)

            print(f"\nSummary saved to: {summary_path}")

        print("=" * 60)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with optional gentle preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  %(prog)s recording.mp3

  # Multiple files
  %(prog)s file1.mp3 file2.wav file3.flac

  # Glob pattern (use quotes!)
  %(prog)s "audio/*.mp3" --output-dir transcripts/

  # Process directory
  %(prog)s --input-dir recordings/ --output-dir transcripts/

  # Recursive directory processing
  %(prog)s --input-dir recordings/ --recursive

  # Multiple output formats
  %(prog)s audio.mp3 --format txt,json,srt

  # Parallel processing
  %(prog)s "*.wav" --threads 4 --output-dir results/

  # Advanced options
  %(prog)s "*.wav" --model large --language en --no-preprocess
        """
    )

    # Input arguments
    parser.add_argument(
        'audio_files',
        nargs='*',
        default=[],
        help='Audio file(s) to transcribe. Supports wildcards like "*.mp3"'
    )

    # Batch processing options
    batch_group = parser.add_argument_group('batch processing')
    batch_group.add_argument(
        '--input-dir',
        help='Directory containing audio files to process'
    )

    batch_group.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process subdirectories recursively (with --input-dir)'
    )

    batch_group.add_argument(
        '--file-pattern',
        default='*',
        help='File pattern to match when using --input-dir (default: "*")'
    )

    batch_group.add_argument(
        '--threads', '-j',
        type=int,
        default=1,
        help='Number of parallel threads for batch processing (default: 1)'
    )

    # Output arguments
    output_group = parser.add_argument_group('output options')
    output_group.add_argument(
        '-o', '--output',
        help='Output transcript file path (only for single file input)',
        default=None
    )

    output_group.add_argument(
        '--output-dir',
        help='Output directory for transcripts (recommended for batch processing)',
        default=None
    )

    output_group.add_argument(
        '--format',
        help='Output format(s): txt,json,srt,vtt (default: txt)',
        default='txt'
    )

    output_group.add_argument(
        '--summary-file',
        help='Save batch processing summary to JSON file',
        default=None
    )

    # Preprocessing options
    preprocessing_group = parser.add_argument_group('preprocessing')
    preprocessing_group.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Skip audio preprocessing (preprocessing is enabled by default)'
    )

    preprocessing_group.add_argument(
        '--save-processed',
        action='store_true',
        help='Save processed audio files (default: delete after transcription)'
    )

    preprocessing_group.add_argument(
        '--noise-reduction',
        type=float,
        default=0.3,  # Much gentler default
        metavar='STRENGTH',
        help='Noise reduction strength (0.0-1.0, default: 0.3)'
    )

    preprocessing_group.add_argument(
        '--voice-boost',
        type=float,
        default=1.5,  # Gentler default
        metavar='DB',
        help='Voice frequency boost in dB (default: 1.5)'
    )

    preprocessing_group.add_argument(
        '--clarity-boost',
        type=float,
        default=1.0,
        metavar='DB',
        help='Clarity frequency boost in dB (default: 1.0)'
    )

    preprocessing_group.add_argument(
        '--presence-boost',
        type=float,
        default=0.5,
        metavar='DB',
        help='High frequency presence boost in dB (default: 0.5)'
    )

    # Transcription options
    transcription_group = parser.add_argument_group('transcription')
    transcription_group.add_argument(
        '--model',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )

    transcription_group.add_argument(
        '--language',
        help='Language code (e.g., en, es, fr). Auto-detect if not specified'
    )

    transcription_group.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (default: 0.0 for deterministic)'
    )

    transcription_group.add_argument(
        '--word-timestamps',
        action='store_true',
        help='Include word-level timestamps in output'
    )

    transcription_group.add_argument(
        '--initial-prompt',
        help='Initial prompt to guide transcription style'
    )

    transcription_group.add_argument(
        '--beam-size',
        type=int,
        help='Beam size for beam search (default: model-dependent)'
    )

    # Logging options
    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    logging_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )

    logging_group.add_argument(
        '--log-file',
        help='Save logs to specified file'
    )

    logging_group.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level (default: INFO)'
    )

    return parser


def main():
    """Main entry point for command line usage."""
    # Check for FFmpeg before parsing arguments
    if not check_ffmpeg_availability():
        sys.exit(1)

    parser = create_parser()
    args = parser.parse_args()

    # Configure logging based on arguments
    setup_logging(args.log_level, args.log_file)

    # Adjust verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Parse output formats
        output_formats = [fmt.strip().lower() for fmt in args.format.split(',')]

        # Validate arguments
        args.format = output_formats  # Replace string with list
        validate_args(args)

        # Collect audio files
        logger.info("Collecting audio files...")
        audio_files = collect_audio_files(args)

        if not audio_files:
            logger.error("No valid audio files found to process")
            sys.exit(1)

        logger.info(f"Found {len(audio_files)} valid audio file(s)")

        # Configure transcription
        transcription_config = TranscriptionConfig(
            model_size=args.model,
            language=args.language,
            temperature=args.temperature,
            word_timestamps=args.word_timestamps,
            verbose=not args.quiet,
            output_formats=output_formats,
            initial_prompt=args.initial_prompt,
            beam_size=args.beam_size
        )

        # Create transcriber
        transcriber = AudioTranscriber(transcription_config)

        # Single file or batch?
        if len(audio_files) == 1 and not args.input_dir:
            # Single file processing
            audio_path = audio_files[0]
            output_path = None

            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / audio_path.stem
            elif args.output:
                output_path = Path(args.output).parent / Path(args.output).stem

            # Configure audio processing
            processing_config = None
            if not args.no_preprocess:
                processing_config = AudioProcessingConfig(
                    noise_reduction_strength=args.noise_reduction,
                    voice_boost_db=args.voice_boost,
                    clarity_boost_db=args.clarity_boost,
                    presence_boost_db=args.presence_boost
                )

            # Show file info
            file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Processing: {audio_path.name} ({file_size:.1f} MB)")

            # Progress indicator for single file
            if not args.quiet:
                def progress_callback(progress, status):
                    # Simple progress bar
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r[{bar}] {progress*100:.1f}% - {status}", end='', flush=True)
                    if progress >= 1.0:
                        print()  # New line when complete
            else:
                progress_callback = None

            # Run transcription
            text, output_file = transcriber.transcribe(
                audio_path=audio_path,
                output_path=output_path,
                preprocess=not args.no_preprocess,
                processing_config=processing_config,
                save_processed=args.save_processed,
                progress_callback=progress_callback
            )

            # Success message
            if not args.quiet:
                print(f"\n✅ Success! Transcript saved to: {output_file}")
                if len(output_formats) > 1:
                    print(f"   Additional formats saved with same base name")
                if args.save_processed and not args.no_preprocess:
                    processed_file = audio_path.parent / f"{audio_path.stem}_processed.wav"
                    print(f"📢 Processed audio saved to: {processed_file}")
        else:
            # Batch processing
            batch_processor = BatchProcessor(transcriber, args)
            summary = batch_processor.process_files(audio_files)
            batch_processor.print_summary(summary)

            # Exit with error code if any files failed
            if summary['failed'] > 0:
                sys.exit(2)  # Partial failure

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        cleanup_temp_files()
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            logger.exception("Full error traceback:")
        cleanup_temp_files()
        sys.exit(1)
    finally:
        # Final cleanup
        cleanup_temp_files()


if __name__ == "__main__":
    main()
