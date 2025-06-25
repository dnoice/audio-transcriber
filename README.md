# 🎙️ Audio Transcriber

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-4.3a.0-green.svg)](https://github.com/yourusername/audio-transcriber)
[![Whisper](https://img.shields.io/badge/Powered%20by-OpenAI%20Whisper-00A67E.svg)](https://github.com/openai/whisper)

A professional-grade audio transcription tool with advanced preprocessing capabilities, batch processing, and multiple output formats. Powered by OpenAI's Whisper model and optimized for challenging audio conditions.

## ✨ Features

- **🎯 Batch Processing**
  - Process multiple files with wildcards (`*.mp3`)
  - Directory processing with optional recursion
  - Preserve folder structure in outputs
  - Individual error handling per file

- **📄 Multiple Output Formats**
  - Plain text (TXT) with metadata
  - Structured JSON with timestamps
  - SubRip subtitles (SRT)
  - WebVTT captions (VTT)

- **🔊 Advanced Audio Preprocessing**
  - Automatic audio normalization
  - Dynamic range compression
  - Advanced noise reduction
  - Voice frequency optimization
  - Adaptive gain control
  - Memory-efficient large file handling

- **📝 High-Quality Transcription**
  - OpenAI Whisper integration
  - Multiple model sizes (tiny to large)
  - 90+ language support
  - Automatic language detection
  - Word-level timestamps

- **🛠️ Professional Features**
  - Progress tracking with time estimates
  - Comprehensive error handling
  - FFmpeg dependency checking
  - Detailed logging options
  - Cross-platform support

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Command Line](#command-line)
  - [Batch Processing](#batch-processing)
  - [Output Formats](#output-formats)
  - [Python API](#python-api)
- [Audio Preprocessing](#-audio-preprocessing)
- [Configuration](#️-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)

#### Installing FFmpeg

**Windows (PowerShell):**

[Use the provided PowerShell installer script](ffmpeg_installer.ps1)

> **Note:** The FFmpeg installer PowerShell script uses a default directory path that might not be suitable for your use case. Make sure to change the directory path to where you want to save the executable and the binary files.

Or download from [ffmpeg.org](https://ffmpeg.org/download.html)

**macOS:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai-whisper pydub noisereduce numpy scipy soundfile
```

## 🎯 Quick Start

### Single File Transcription

```bash
# Basic usage with preprocessing
python audio_transcribe.py interview.mp3

# Skip preprocessing for clean audio
python audio_transcribe.py podcast.mp3 --no-preprocess

# Use larger model for better accuracy
python audio_transcribe.py lecture.wav --model large

# Multiple output formats
python audio_transcribe.py meeting.mp3 --format txt,json,srt
```

### Batch Processing

```bash
# Process all MP3 files in current directory
python audio_transcribe.py "*.mp3" --output-dir transcripts/

# Process entire directory recursively
python audio_transcribe.py --input-dir recordings/ --recursive

# Multiple files with specific model
python audio_transcribe.py file1.mp3 file2.wav file3.flac --model medium
```

## 📖 Usage

### Command Line

```bash
python audio_transcribe.py [audio_files...] [options]
```

#### Input Options

| Option | Description | Example |
|--------|-------------|---------|
| `audio_files` | Audio file(s) to transcribe | `audio.mp3` or `"*.wav"` |
| `--input-dir` | Process all audio files in directory | `--input-dir recordings/` |
| `--recursive` | Include subdirectories | `--recursive` |
| `--file-pattern` | Pattern for directory search | `--file-pattern "interview_*"` |

#### Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output path (single file only) | `[input]_transcript.txt` |
| `--output-dir` | Output directory for all files | Current directory |
| `--format` | Output format(s): txt,json,srt,vtt | `txt` |

#### Preprocessing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-preprocess` | Skip audio preprocessing | Preprocessing enabled |
| `--save-processed` | Keep processed audio files | Delete after transcription |
| `--noise-reduction` | Noise reduction strength (0.0-1.0) | `0.8` |
| `--voice-boost` | Voice frequency boost in dB | `3.0` |

#### Transcription Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model size: tiny/base/small/medium/large | `base` |
| `--language` | Language code (e.g., en, es, fr) | Auto-detect |
| `--temperature` | Sampling temperature | `0.0` |
| `--word-timestamps` | Include word-level timestamps | Disabled |

#### Other Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable detailed logging |
| `-q, --quiet` | Suppress non-essential output |
| `-h, --help` | Show help message |

### Batch Processing Examples

The tool excels at processing multiple files efficiently:

```bash
# Process with glob pattern (always use quotes!)
python audio_transcribe.py "audio/*.mp3" --output-dir results/

# Process directory with specific pattern
python audio_transcribe.py --input-dir podcasts/ --file-pattern "episode_*"

# Recursive processing with multiple formats
python audio_transcribe.py --input-dir interviews/ --recursive --format json,srt

# Advanced batch processing
python audio_transcribe.py "2024_recordings/*.wav" \
    --model large \
    --language en \
    --format txt,json,vtt \
    --output-dir transcripts/2024/
```

### Output Formats

#### Text Format (TXT)

Plain text transcription with metadata:

```text
This is the transcribed text content...

==================================================
# Transcription Metadata
# Language: en
# Duration: 125.45 seconds
# Segments: 24
```

#### JSON Format

Structured data with full timestamps:

```json
{
  "text": "This is the transcribed text...",
  "language": "en",
  "duration": 125.45,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "This is the transcribed text"
    }
  ]
}
```

#### SRT Format

Standard subtitle format:

```srt
1
00:00:00,000 --> 00:00:02,500
This is the transcribed text

2
00:00:02,500 --> 00:00:05,000
Second segment of speech
```

#### VTT Format

WebVTT for HTML5 video:

```vtt
WEBVTT

00:00:00.000 --> 00:00:02.500
This is the transcribed text

00:00:02.500 --> 00:00:05.000
Second segment of speech
```

### Python API

#### Basic Usage

```python
from audio_transcribe import AudioTranscriber

# Create transcriber
transcriber = AudioTranscriber()

# Transcribe single file
text, output_path = transcriber.transcribe("interview.mp3")
print(f"Transcription saved to: {output_path}")
```

#### Advanced Configuration

```python
from audio_transcribe import (
    AudioTranscriber,
    TranscriptionConfig,
    AudioProcessingConfig
)

# Configure audio processing
processing_config = AudioProcessingConfig(
    normalize=True,
    compress_dynamics=True,
    reduce_noise=True,
    noise_reduction_strength=0.9,
    voice_boost_db=4.0,
    clarity_boost_db=3.0
)

# Configure transcription
transcription_config = TranscriptionConfig(
    model_size="medium",
    language="en",
    temperature=0.0,
    word_timestamps=True,
    output_formats=["txt", "json", "srt"]
)

# Create transcriber
transcriber = AudioTranscriber(transcription_config)

# Transcribe with custom settings
text, output_path = transcriber.transcribe(
    "podcast.mp3",
    output_path="transcripts/podcast",
    preprocess=True,
    processing_config=processing_config,
    save_processed=True
)
```

#### Batch Processing via API

```python
from pathlib import Path
from audio_transcribe import AudioTranscriber, TranscriptionConfig

# Configure for JSON output
config = TranscriptionConfig(
    model_size="small",
    output_formats=["json", "srt"]
)

transcriber = AudioTranscriber(config)

# Process directory
audio_dir = Path("recordings")
output_dir = Path("transcripts")
output_dir.mkdir(exist_ok=True)

for audio_file in audio_dir.glob("*.mp3"):
    print(f"Processing {audio_file.name}...")

    try:
        text, output_path = transcriber.transcribe(
            audio_file,
            output_path=output_dir / audio_file.stem,
            preprocess=True
        )
        print(f"✓ Completed: {output_path}")
    except Exception as e:
        print(f"✗ Failed: {e}")
```

#### Audio Processing Only

```python
from audio_transcribe import AudioProcessor, AudioProcessingConfig

# Configure processing
config = AudioProcessingConfig(
    noise_reduction_strength=0.95,
    voice_boost_db=6.0,
    target_level=0.4
)

# Process audio
processor = AudioProcessor("noisy_recording.mp3", config)
processed_path = processor.process("clean_recording.wav")
```

## 🔊 Audio Preprocessing

The preprocessing pipeline optimizes audio for transcription:

### 1. **Normalization**

- Brings audio to optimal levels
- Prevents clipping
- Ensures consistent input levels

### 2. **Dynamic Range Compression**

- Evens out volume variations
- Makes quiet parts audible
- Controls loud peaks

### 3. **Noise Reduction**

- Advanced spectral gating
- Preserves voice while removing background noise
- Adaptive to different noise types

### 4. **Voice Frequency Optimization**

- **Fundamentals (85-300 Hz)**: Enhanced voice body
- **Clarity (1-4 kHz)**: Improved intelligibility
- **Presence (5+ kHz)**: Added brilliance

### 5. **Adaptive Gain Control**

- Maintains consistent volume
- Sample-rate aware processing
- Smooth transitions

## ⚙️ Configuration

### Audio Processing Parameters

Configure audio preprocessing with these parameters:

```python
AudioProcessingConfig(
    normalize=True,                    # Apply normalization
    compress_dynamics=True,            # Apply compression
    reduce_noise=True,                 # Apply noise reduction
    optimize_voice=True,               # Optimize voice frequencies
    adaptive_gain=True,                # Apply adaptive gain
    noise_reduction_strength=0.8,      # 0.0-1.0 (higher = more reduction)
    compression_threshold=-20.0,       # dB threshold for compression
    compression_ratio=4.0,             # Compression ratio
    voice_boost_db=3.0,               # Boost for fundamental frequencies
    clarity_boost_db=2.0,             # Boost for clarity frequencies
    presence_boost_db=1.0,            # Boost for presence frequencies
    target_level=0.3                  # Target level for adaptive gain
)
```

### Transcription Parameters

Configure transcription behavior with these settings:

```python
TranscriptionConfig(
    model_size="base",                 # Model: tiny/base/small/medium/large
    language=None,                     # Language code or None for auto-detect
    temperature=0.0,                   # Sampling temperature (0.0 = deterministic)
    compression_ratio_threshold=2.4,   # Threshold for failed decoding
    logprob_threshold=-1.0,           # Average log probability threshold
    no_speech_threshold=0.6,          # No speech probability threshold
    condition_on_previous_text=True,  # Use previous text as context
    word_timestamps=False,            # Include word-level timestamps
    verbose=True,                     # Show progress
    output_formats=["txt"]            # Output formats list
)
```

## 📚 Examples

### Interview Transcription with Subtitles

```bash
python audio_transcribe.py interview.mp3 \
    --model medium \
    --format txt,srt,vtt \
    --save-processed
```

### Batch Process Podcast Episodes

```bash
python audio_transcribe.py --input-dir "podcasts/season1/" \
    --recursive \
    --model small \
    --language en \
    --format json \
    --output-dir "transcripts/season1/"
```

### Noisy Recording Enhancement

```bash
python audio_transcribe.py "field_recording_*.wav" \
    --noise-reduction 0.95 \
    --voice-boost 6.0 \
    --save-processed \
    --output-dir cleaned/
```

### Meeting Minutes with Timestamps

```python
from audio_transcribe import AudioTranscriber, TranscriptionConfig

config = TranscriptionConfig(
    model_size="medium",
    word_timestamps=True,
    output_formats=["json"]
)

transcriber = AudioTranscriber(config)

# Process and get structured data
text, json_path = transcriber.transcribe(
    "meeting_recording.mp3",
    output_path="meeting_minutes"
)

# Load and process the JSON for custom formatting
import json
with open(json_path) as f:
    data = json.load(f)

    for segment in data["segments"]:
        timestamp = f"[{segment['start']:.1f}s]"
        print(f"{timestamp} {segment['text']}")
```

## 🔧 Troubleshooting

### Common Issues

#### FFmpeg Not Found

```bash
# Verify installation
ffmpeg -version

# The script will show installation instructions if FFmpeg is missing
```

#### Memory Issues with Large Files

- Files over 500MB are automatically processed efficiently
- Use smaller models for very long recordings
- The script handles memory management automatically

#### Poor Transcription Quality

1. Enable preprocessing (default): Remove `--no-preprocess`
2. Try a larger model: `--model medium` or `--model large`
3. Adjust enhancement: `--noise-reduction 0.9 --voice-boost 5.0`
4. Specify language: `--language en`

#### Batch Processing Errors

- Individual file errors won't stop the batch
- Check the summary for failed files
- Use `--verbose` for detailed error messages

### Performance Tips

**Model Selection Guide:**

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | 74 MB | ★★★★★ | ★ | Quick drafts |
| base | 142 MB | ★★★★ | ★★ | General use |
| small | 466 MB | ★★★ | ★★★ | Better accuracy |
| medium | 1.5 GB | ★★ | ★★★★ | Professional |
| large | 2.9 GB | ★ | ★★★★★ | Best quality |

**Processing Speed:**

- First run downloads the model
- Preprocessing adds ~10-30% to processing time
- Batch processing is more efficient than individual files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install pytest black flake8  # Development tools

# Run tests
pytest tests/

# Format code
black audio_transcribe.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - State-of-the-art transcription
- [PyDub](https://github.com/jiaaro/pydub) - Audio manipulation
- [noisereduce](https://github.com/timsainb/noisereduce) - Noise reduction algorithms
- [SciPy](https://scipy.org/) - Signal processing
- All contributors and users of this project

## 🌐 Supported Languages

OpenAI Whisper supports over 90 languages with varying levels of accuracy. Here are some of the most commonly used:

**Excellent Support** (>90% accuracy):

- English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian

**Very Good Support** (>85% accuracy):

- Japanese, Chinese, Korean, Arabic, Turkish, Indonesian, Vietnamese, Thai

**Good Support** (>80% accuracy):

- Hindi, Swedish, Finnish, Norwegian, Danish, Greek, Czech, Romanian, Hungarian

**Additional Languages**:

- Hebrew, Persian, Ukrainian, Croatian, Slovak, Catalan, Malay, and 60+ more

### 📊 Language Performance Metrics

![Whisper Language Performance](assets/whisper-language-performance.svg)
*Performance comparison of Whisper large-v3 and large-v2 models across languages using Word Error Rate (WER) and Character Error Rate (CER*) on Common Voice 15 and FLEURS datasets. Lower values indicate better performance. Click to zoom for detailed metrics.*

For a complete list and accuracy metrics, see the [Whisper language documentation](https://github.com/openai/whisper#available-models-and-languages).

## 🚦 System Requirements

- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 10GB disk space (for large model)
- **FFmpeg**: Required for audio processing
- **Python**: 3.8 or higher

---

## Made with ❤️ by dendogg

*If you find this project helpful, please consider giving it a ⭐!*
