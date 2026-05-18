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

> **Language & noise handling:** Both scripts are configured to transcribe in Portuguese (`pt`) by default, avoiding misdetection caused by instrumental music or noise at the beginning/end of recordings. A VAD (Voice Activity Detection) filter is also applied to suppress non-speech segments such as background noise and music, improving transcription accuracy.

## Requirements

- Python 3.9+
- NVIDIA GPU (optional, but recommended for faster processing)
  - [CUDA Toolkit 12](https://developer.nvidia.com/cuda-toolkit-archive) — see below how to identify the right version for your hardware
  - [cuBLAS and cuDNN DLLs](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs) — download `cuBLAS.and.cuDNN_CUDA12_win_v3.7z` (Purfview package for Windows)

### Identifying the right CUDA version for your hardware

Run the following command to check your driver and the maximum supported CUDA version:

```bash
nvidia-smi
```

Look for the **CUDA Version** field in the top-right corner of the output:

```
| NVIDIA-SMI 596.36   Driver Version: 596.36   CUDA Version: 13.2 |
```

> This value indicates the **maximum CUDA version your driver supports** — it does not mean the CUDA Toolkit is installed, only that your driver is compatible up to that version.

Since faster-whisper requires **CUDA 12**, check if your driver supports it:

- `CUDA Version >= 12.0` → you can install CUDA Toolkit 12 ✅
- `CUDA Version < 12.0` → you need to update your NVIDIA driver first before proceeding ❌

On the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) page, select any **12.x** version and the Windows installer. The latest 12.x release is recommended.

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
python transcribe.py
```

### `transcribe_batch.py`

Set `AUDIO_DIR` and `OUTPUT_DIR` at the top of the script, then run:

```bash
cd C:\path\to\Transcribe
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
