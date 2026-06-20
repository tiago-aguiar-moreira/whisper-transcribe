# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two Python scripts that transcribe `.m4a` audio files to `.txt` using `faster-whisper`. Output is plain text with 5-minute time markers, optimized for LLM consumption.

## Running the scripts

Both scripts must be run **from the directory containing the audio files**, not from this repository. They use `os.getcwd()` to locate inputs and write outputs.

```bash
cd "path\to\audio\files"
S:\workspace\whisper-transcribe\.venv\Scripts\activate
python S:\workspace\whisper-transcribe\transcribe.py       # interactive, one file
python S:\workspace\whisper-transcribe\transcribe_batch.py # batch, all pending
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install faster-whisper
```

FFmpeg is bundled via PyAV — no separate install needed. GPU support requires CUDA Toolkit 12 + cuBLAS/cuDNN DLLs.

## Architecture

Both scripts share the same constants, `get_output_filename()`, and `transcribe()` function — they are intentionally duplicated (no shared module).

**`transcribe.py`** — interactive: lists `.m4a` files with `[✓]` for already-transcribed, prompts user to pick one by number, asks confirmation before overwriting.

**`transcribe_batch.py`** — batch: auto-skips already-transcribed files, loads the model once, processes all pending files sequentially.

### File naming convention

`get_output_filename()` strips the `-new` suffix before generating the output name:

| Input | Output |
|---|---|
| `20260516-1.m4a` | `20260516-1-transcript.txt` |
| `20260516-1-new.m4a` | `20260516-1-transcript.txt` |

### Output format

```
Idioma: pt (98.50%)

[0min] First segment text...
[5min] Next segment text...
```

## Configuration constants

Defined at the top of each script:

| Constant | Default | Notes |
|---|---|---|
| `MODEL_SIZE` | `small` | `tiny`, `small`, `medium`, `large-v3`, `turbo` |
| `DEVICE` | `cpu` | `cuda` for GPU |
| `COMPUTE_TYPE` | `int8` | `float16` or `int8_float16` for GPU |
| `LANGUAGE` | `pt` | Fixed to avoid misdetection from intro music/noise |
| `TIME_MARK_INTERVAL` | `300` | Seconds between time markers |
| `CHUNK_LENGTH` | `30` | Audio chunk size in seconds (limits RAM usage) |

VAD filter (`vad_filter=True`, `min_silence_duration_ms=500`) suppresses background noise and music. Language is fixed to `pt` because instrumental intros can cause wrong language detection.
