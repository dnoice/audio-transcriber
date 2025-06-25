#!/usr/bin/env python3
"""
Title: Audio Transcriber
File name: audio_transcribe.py
Author: Dennis 'dendogg' Smaltz
Version: 2.1b.0
Date: 2023-10-01
Description: This script provides a command line interface for transcribing audio files
             with optional audio enhancement features such as noise reduction, dynamic
             range compression, and voice frequency enhancement. It supports various
             audio formats and can output transcripts in text files. The tool is designed
             to be user-friendly, with sensible defaults and extensive logging for debugging.
License: MIT License
Requirements: openai-whisper, pydub, noisereduce, numpy, scipy, soundfile
Usage: python audio_transcribe.py <audio_file> [options]
Example: python audio_transcribe.py recording.mp3 --output transcript.txt --no-enhance
Notes:
- Ensure you have the required libraries installed:
  pip install openai-whisper pydub noisereduce numpy scipy soundfile
- For best results, use high-quality audio files.
- The script supports various audio formats including MP3, WAV, FLAC, and OGG.
- The Whisper model can be specified with the --model option (default is 'base').
- Audio enhancement can be disabled with the --no-enhance option.
- The script can automatically detect the language of the audio or use a specified language.
- Verbose output can be enabled with the --verbose option for detailed logging.
- The output directory can be specified with the --output-dir option.
- The script handles both single-file transcription and batch processing of multiple files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

import numpy as np
import whisper
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancementConfig:
    """Configuration for audio enhancement parameters."""

    normalize: bool = True
    compress_dynamics: bool = True
    reduce_noise: bool = True
    enhance_voice: bool = True
    adaptive_gain: bool = True
    noise_reduction_strength: float = 0.8
    compression_threshold: float = -20.0
    compression_ratio: float = 4.0
    voice_boost_db: float = 3.0
    clarity_boost_db: float = 2.0
    presence_boost_db: float = 1.0
    target_level: float = 0.3


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


class AudioEnhancer:
    """
    Audio enhancement class for improving voice clarity.

    This class provides various audio processing techniques to enhance
    speech quality, particularly useful for quiet or noisy recordings.

    Attributes:
        audio_path (Path): Path to the input audio file
        config (EnhancementConfig): Enhancement configuration parameters
    """

    def __init__(self, audio_path: str, config: Optional[EnhancementConfig] = None):
        """
        Initialize the AudioEnhancer.

        Args:
            audio_path: Path to the input audio file
            config: Enhancement configuration (uses defaults if None)
        """
        self.audio_path = Path(audio_path)
        self.config = config or EnhancementConfig()
        self._validate_input()

    def _validate_input(self) -> None:
        """Validate that the input file exists and is readable."""
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        if not self.audio_path.is_file():
            raise ValueError(f"Path is not a file: {self.audio_path}")

    def enhance(self, output_path: Optional[str] = None) -> str:
        """
        Apply audio enhancement pipeline.

        Args:
            output_path: Optional path for enhanced audio output

        Returns:
            Path to the enhanced audio file

        Raises:
            RuntimeError: If enhancement fails
        """
        if output_path is None:
            output_path = self.audio_path.stem + "_enhanced.wav"

        output_path = Path(output_path)

        try:
            logger.info("Starting audio enhancement pipeline...")

            # Load audio
            audio = self._load_audio()

            # Apply enhancement pipeline
            if self.config.normalize:
                logger.info("Normalizing audio levels...")
                audio = normalize(audio)

            if self.config.compress_dynamics:
                logger.info("Applying dynamic range compression...")
                audio = compress_dynamic_range(
                    audio,
                    threshold=self.config.compression_threshold,
                    ratio=self.config.compression_ratio
                )

            # Convert to numpy for advanced processing
            samples, sample_rate = self._audio_to_numpy(audio)

            if self.config.reduce_noise:
                logger.info("Reducing background noise...")
                samples = self._reduce_noise(samples, sample_rate)

            if self.config.enhance_voice:
                logger.info("Enhancing voice frequencies...")
                samples = self._enhance_voice_frequencies(samples, sample_rate)

            if self.config.adaptive_gain:
                logger.info("Applying adaptive gain control...")
                samples = self._adaptive_gain_control(samples)

            # Final limiting
            samples = np.clip(samples, -1.0, 1.0)

            # Save enhanced audio
            sf.write(str(output_path), samples, sample_rate)
            logger.info(f"Enhanced audio saved to: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}")
            raise RuntimeError(f"Audio enhancement failed: {str(e)}")

    def _load_audio(self) -> AudioSegment:
        """Load audio file using pydub."""
        try:
            return AudioSegment.from_file(str(self.audio_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")

    def _audio_to_numpy(self, audio: AudioSegment) -> Tuple[np.ndarray, int]:
        """
        Convert AudioSegment to numpy array.

        Args:
            audio: AudioSegment object

        Returns:
            Tuple of (samples as numpy array, sample rate)
        """
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sample_rate = audio.frame_rate

        # Normalize to [-1, 1]
        max_val = float(2 ** (audio.sample_width * 8 - 1))
        samples = samples / max_val

        # Convert stereo to mono if necessary
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)

        return samples, sample_rate

    def _reduce_noise(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction using spectral gating.

        Args:
            samples: Audio samples
            sample_rate: Sample rate

        Returns:
            Noise-reduced samples
        """
        try:
            return nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                prop_decrease=self.config.noise_reduction_strength,
                stationary=False
            )
        except Exception as e:
            logger.warning(f"Advanced noise reduction failed: {e}. Using fallback.")
            return self._simple_noise_gate(samples)

    def _simple_noise_gate(self, samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Simple noise gate as fallback."""
        mask = np.abs(samples) > threshold
        return samples * mask

    def _enhance_voice_frequencies(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance frequencies typical of human voice.

        Targets:
        - Fundamental frequencies: 85-300 Hz
        - Formants/clarity: 1-4 kHz
        - Presence: 5+ kHz

        Args:
            samples: Audio samples
            sample_rate: Sample rate

        Returns:
            Voice-enhanced samples
        """
        # Remove low-frequency rumble
        samples = self._highpass_filter(samples, 80, sample_rate)

        # Boost voice fundamentals
        samples = self._bandpass_boost(
            samples, 100, 300, sample_rate,
            boost_db=self.config.voice_boost_db
        )

        # Boost clarity frequencies
        samples = self._bandpass_boost(
            samples, 1000, 4000, sample_rate,
            boost_db=self.config.clarity_boost_db
        )

        # Add presence
        samples = self._highshelf_filter(
            samples, 5000, sample_rate,
            boost_db=self.config.presence_boost_db
        )

        return samples

    def _highpass_filter(self, data: np.ndarray, cutoff: float,
                        fs: int, order: int = 5) -> np.ndarray:
        """Apply Butterworth high-pass filter."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        if normal_cutoff >= 1.0:
            return data

        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    def _bandpass_boost(self, data: np.ndarray, lowcut: float, highcut: float,
                       fs: int, boost_db: float = 3.0, order: int = 5) -> np.ndarray:
        """Boost specific frequency band."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        if low >= 1.0 or high >= 1.0 or low >= high:
            return data

        b, a = butter(order, [low, high], btype='band')
        band_data = lfilter(b, a, data)

        boost_factor = 10 ** (boost_db / 20)
        return data + (band_data * (boost_factor - 1))

    def _highshelf_filter(self, data: np.ndarray, cutoff: float,
                         fs: int, boost_db: float = 2.0) -> np.ndarray:
        """Apply high-shelf filter for presence."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        if normal_cutoff >= 1.0:
            return data

        b, a = butter(2, normal_cutoff, btype='high', analog=False)
        high_freqs = lfilter(b, a, data)

        boost_factor = 10 ** (boost_db / 20)
        return data + (high_freqs * (boost_factor - 1))

    def _adaptive_gain_control(self, samples: np.ndarray,
                              frame_length: int = 1024) -> np.ndarray:
        """
        Apply adaptive gain control for consistent volume.

        Args:
            samples: Audio samples
            frame_length: Length of frames for processing

        Returns:
            Gain-controlled samples
        """
        output = np.zeros_like(samples)
        target_level = self.config.target_level

        for i in range(0, len(samples) - frame_length, frame_length):
            frame = samples[i:i + frame_length]

            # Calculate frame energy
            energy = np.sqrt(np.mean(frame ** 2))

            if energy > 0.001:  # Avoid division by zero
                # Calculate and limit gain
                gain = min(target_level / energy, 10.0)

                # Smooth gain changes
                if i > 0:
                    prev_energy = np.sqrt(np.mean(
                        samples[i-frame_length:i] ** 2
                    ))
                    if prev_energy > 0.001:
                        prev_gain = target_level / prev_energy
                        gain = 0.7 * gain + 0.3 * prev_gain

                output[i:i + frame_length] = frame * gain
            else:
                output[i:i + frame_length] = frame

        # Handle remaining samples
        remaining = len(samples) % frame_length
        if remaining > 0:
            output[-remaining:] = samples[-remaining:]

        return output


class AudioTranscriber:
    """
    Main transcription class using OpenAI's Whisper.

    Handles the transcription process with optional audio enhancement
    and various output formats.

    Attributes:
        config (TranscriptionConfig): Transcription configuration
        model: Loaded Whisper model
    """

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Initialize the AudioTranscriber.

        Args:
            config: Transcription configuration (uses defaults if None)
        """
        self.config = config or TranscriptionConfig()
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        if self.model is None:
            logger.info(f"Loading Whisper {self.config.model_size} model...")
            self.model = whisper.load_model(self.config.model_size)
            logger.info("Model loaded successfully")

    def transcribe(self, audio_path: str, output_path: Optional[str] = None,
                  enhance: bool = True,
                  enhancement_config: Optional[EnhancementConfig] = None,
                  save_enhanced: bool = True) -> Tuple[str, str]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to input audio file
            output_path: Optional path for output text file
            enhance: Whether to enhance audio before transcription
            enhancement_config: Optional enhancement configuration
            save_enhanced: Whether to keep the enhanced audio file

        Returns:
            Tuple of (transcribed text, output file path)

        Raises:
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If transcription fails
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if output_path is None:
            output_path = audio_path.stem + "_transcript.txt"

        output_path = Path(output_path)

        # Load model if not already loaded
        self.load_model()

        # Process audio
        processing_path = str(audio_path)
        enhanced_path = None

        if enhance:
            enhancer = AudioEnhancer(str(audio_path), enhancement_config)
            enhanced_path = enhancer.enhance()
            processing_path = enhanced_path

        try:
            # Transcribe
            logger.info("Starting transcription...")
            result = self._transcribe_audio(processing_path)

            # Save transcript
            self._save_transcript(result, str(output_path))

            # Clean up if requested
            if enhance and not save_enhanced and enhanced_path:
                Path(enhanced_path).unlink()
                logger.info("Temporary enhanced audio removed")

            # Log results
            self._log_results(result, str(output_path), enhanced_path if save_enhanced else None)

            return result["text"].strip(), str(output_path)

        except Exception as e:
            # Clean up on error
            if enhance and enhanced_path and Path(enhanced_path).exists():
                Path(enhanced_path).unlink()

            logger.error(f"Transcription failed: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")

    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Execute the Whisper transcription."""
        return self.model.transcribe(
        audio_path,
        language=self.config.language,
        task='transcribe',
        temperature=self.config.temperature,
        compression_ratio_threshold=self.config.compression_ratio_threshold,
        logprob_threshold=self.config.logprob_threshold,
        no_speech_threshold=self.config.no_speech_threshold,
        condition_on_previous_text=self.config.condition_on_previous_text,
        initial_prompt=self.config.initial_prompt,
        word_timestamps=self.config.word_timestamps,
        prepend_punctuations="\"'([{-",
        append_punctuations="\"'.。,，!！?？:：)]}、",
        verbose=self.config.verbose
    )

    def _save_transcript(self, result: Dict[str, Any], output_path: str) -> None:
        """Save transcription results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result["text"].strip())

    def _log_results(self, result: Dict[str, Any], output_path: str,
                    enhanced_path: Optional[str]) -> None:
        """Log transcription results."""
        logger.info("=" * 50)
        logger.info("Transcription completed successfully!")
        logger.info(f"Transcript saved to: {output_path}")

        if enhanced_path:
            logger.info(f"Enhanced audio saved to: {enhanced_path}")

        logger.info(f"Detected language: {result.get('language', 'Unknown')}")
        logger.info(f"Duration: {result.get('duration', 0):.2f} seconds")
        logger.info(f"Number of segments: {len(result.get('segments', []))}")

        # Preview
        text = result["text"].strip()
        preview_length = 500
        if len(text) > preview_length:
            preview = text[:preview_length] + "..."
        else:
            preview = text

        logger.info("-" * 50)
        logger.info("Transcript preview:")
        logger.info(preview)
        logger.info("=" * 50)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with optional enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.mp3
  %(prog)s recording.mp3 -o transcript.txt
  %(prog)s recording.mp3 --no-enhance
  %(prog)s recording.mp3 --model large --language en
  %(prog)s recording.mp3 --save-enhanced --output-dir results/
        """
    )

    # Positional arguments
    parser.add_argument(
        'audio_file',
        help='Path to input audio file'
    )

    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output transcript file path',
        default=None
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory for all files',
        default=None
    )

    # Enhancement options
    enhancement_group = parser.add_argument_group('enhancement options')
    enhancement_group.add_argument(
        '--no-enhance',
        action='store_true',
        help='Skip audio enhancement (by default, enhancement is enabled)'
    )

    enhancement_group.add_argument(
        '--save-enhanced',
        action='store_true',
        help='Save enhanced audio file'
    )

    enhancement_group.add_argument(
        '--noise-reduction',
        type=float,
        default=0.8,
        help='Noise reduction strength (0.0-1.0, default: 0.8)'
    )

    # Transcription options
    transcription_group = parser.add_argument_group('transcription options')
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

    # Logging options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress non-essential output'
    )

    return parser


def main():
    """Main entry point for command line usage."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Prepare output paths
    audio_path = Path(args.audio_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output:
            output_path = output_dir / args.output
        else:
            output_path = output_dir / (audio_path.stem + "_transcript.txt")
    else:
        if args.output:
            output_path = args.output
    # Configure enhancement only if enabled
    enhancement_config = None
    if not args.no_enhance:
        enhancement_config = EnhancementConfig(
            noise_reduction_strength=args.noise_reduction
        )
    enhancement_config = EnhancementConfig(
        noise_reduction_strength=args.noise_reduction
    )

    # Configure transcription
    transcription_config = TranscriptionConfig(
        model_size=args.model,
        language=args.language,
        verbose=not args.quiet
    )

    # Run transcription
    try:
        transcriber = AudioTranscriber(transcription_config)
        text, output_file = transcriber.transcribe(
            audio_path=str(audio_path),
            output_path=str(output_path) if output_path else None,
            enhance=not args.no_enhance,
            enhancement_config=enhancement_config,
            save_enhanced=args.save_enhanced
        )

        if not args.quiet:
            print(f"\nSuccess! Transcript saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
