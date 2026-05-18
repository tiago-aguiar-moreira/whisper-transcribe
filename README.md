# Transcribe

Python scripts for automatic audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper), optimized for LLM consumption.

## Scripts

### `transcribe.py`
Interactive script for transcribing a single audio file.

- Lists all `.m4a` files found in the current directory
- Indicates which files have already been transcribed
- Prompts the user to select a file by number
- Asks for confirmation before overwriting an existing transcription
- Saves the output `.txt` in the same directory where the script is executed

### `transcribe_batch.py`
Batch script for transcribing all audio files in a configured directory.

- Skips files that have already been transcribed
- Loads the model once and reuses it across all files
- Saves output `.txt` files to the configured output directory

## Output format

Both scripts generate a plain text file with time markers every 5 minutes, optimized for LLM consumption:

```
Language detected: pt (98.50%)

[0min] Lorem ipsum dolor sit amet...
[5min] Consectetur adipiscing elit...
[10min] Sed do eiusmod tempor...
```

## Requirements

- Python 3.9+
- NVIDIA GPU (optional, but recommended for faster processing)
  - CUDA Toolkit 12
  - cuBLAS and cuDNN DLLs ([Purfview package for Windows](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs))

## Setup

**1. Create and activate the virtual environment**

```bash
cd C:\path\to\Transcribe
python -m venv .venv
.venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install faster-whisper
```

> FFmpeg does not need to be installed separately. The PyAV library includes it.

## Running

### `transcribe.py`

Navigate to the directory containing the audio files, then run the script:

```bash
cd "E:\path\to\audio\files"
.venv\Scripts\activate
python C:\Users\tiago\workspace\Transcribe\transcribe.py
```

### `transcribe_batch.py`

Set `AUDIO_DIR` and `OUTPUT_DIR` at the top of the script, then run:

```bash
cd C:\Users\tiago\workspace\Transcribe
.venv\Scripts\activate
python transcribe_batch.py
```

## Configuration

Both scripts expose constants at the top of the file for easy customization:

| Constant | Description | Default |
|---|---|---|
| `MODEL_SIZE` | Whisper model to use (`tiny`, `small`, `medium`, `large-v3`, `turbo`) | `small` |
| `DEVICE` | Inference device (`cpu` or `cuda`) | `cpu` |
| `COMPUTE_TYPE` | Numeric precision (`int8`, `float16`, `int8_float16`) | `int8` |
| `LANGUAGE` | Audio language code | `pt` |
| `TIME_MARK_INTERVAL` | Time marker interval in seconds | `300` (5 min) |
