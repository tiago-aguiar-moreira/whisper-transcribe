# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Two Python scripts that transcribe audio files (`AUDIO_EXTENSIONS`) using `faster-whisper`. The `.txt` (plain text with 5-minute time markers, optimized for LLM consumption) is always written; `--srt` and `--words` select the extra outputs (`.srt` subtitle, `.words.json` with per-word timestamps). With neither flag, both are generated.

## Running the scripts

Both scripts must be run **from the directory containing the audio files**, not from this repository. They use `os.getcwd()` to locate inputs and write outputs.

```bash
cd "path\to\audio\files"
S:\workspace\whisper-transcribe\.venv\Scripts\activate
python S:\workspace\whisper-transcribe\transcribe.py       # interactive, one file
python S:\workspace\whisper-transcribe\transcribe_batch.py # batch, all pending
python S:\workspace\whisper-transcribe\transcribe_batch.py --srt   # .txt + .srt only (--words: .txt + .words.json)
```

`transcribe_batch.bat` is a Windows convenience launcher for the batch script: auto-creates the `.venv`, runs `pip install -r requirements.txt` (pinned versions), then runs `transcribe_batch.py` forwarding its arguments (`%*`). Call it from the audio directory the same way:

```bat
cd "path\to\audio\files"
"S:\workspace\whisper-transcribe\transcribe_batch.bat"
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `faster-whisper==1.2.1` and `av<18` (PyAV 19 removed `metadata_errors`, still used by faster-whisper 1.2.1).

FFmpeg is bundled via PyAV — no separate install needed. GPU support requires CUDA Toolkit 12 + cuBLAS/cuDNN DLLs.

## Architecture

Both scripts share the same block — constants, `parse_args()`, `get_output_files()`, the formatters (`build_transcript`, `format_txt`, `format_srt`, `format_words_json`), `write_outputs()`, `transcribe()` and `RunLog` — intentionally duplicated (no shared module). Only the docstring and `main()` differ; keep the shared block identical in both files.

Behaviour common to both:
- **Atomic writes:** every output is written to `<file>.tmp` and then `os.replace`d, with the `.txt` last; an exception or Ctrl+C deletes the `.tmp` files, so a partial transcript never counts as done.
- **Done = all requested outputs exist** (`is_done`). Asking for an output a file doesn't have yet (e.g. `.words.json`) makes it pending again and it is re-transcribed.
- **Run log:** `logs\AAAA-MM-DD_HH-MM-SS.log` in the repository folder (git-ignored); console mirrored with a timestamp per line; tracebacks go only to the file (`RunLog.detail`).
- **Console progress** every `PROGRESS_INTERVAL` seconds of audio (`... 15min de 72min`) instead of printing the text.
- **Exit codes:** 0 ok, 1 some file failed (batch), 2 fatal error / failure, 130 interrupted.
- `word_timestamps=True` only when `.words.json` is requested.

**`transcribe.py`** — interactive: lists audio files with `[✓]` when all requested outputs exist, prompts user to pick one by number, asks confirmation before overwriting any existing output.

**`transcribe_batch.py`** — batch: auto-skips already-transcribed files, loads the model once, processes all pending files sequentially. A failure in one file is logged and the batch continues; the summary lists failures.

### File naming convention

`get_output_files()` strips the `-new` suffix (only at the end of the name) before generating the output names:

| Input | Output |
|---|---|
| `20260516-1.m4a` | `20260516-1-transcript.txt` / `.srt` / `.words.json` |
| `20260516-1-new.m4a` | `20260516-1-transcript.txt` / `.srt` / `.words.json` |
| `entrevista-newton.m4a` | `entrevista-newton-transcript.txt` (not stripped) |

### Output format

```
Idioma: pt (98.50%)

[0min] First segment text...
[5min] Next segment text...
```

`.srt`: one SubRip block per Whisper segment. `.words.json`: `{version, language, language_probability, source, duration, segments: [{start, end, text, words: [{start, end, word, probability}]}]}`, one word per line; words are kept in order, without overlap, clamped to `duration`, and pieces without a leading space are joined to the previous word (`bem` + `-vindo`). Changing this format requires bumping `JSON_FORMAT_VERSION`.

## Configuration constants

Defined at the top of each script:

| Constant | Default | Notes |
|---|---|---|
| `MODEL_SIZE` | `turbo` | `tiny`, `small`, `medium`, `large-v3`, `turbo` |
| `DEVICE` | `cuda` | `cpu` without GPU |
| `COMPUTE_TYPE` | `float16` | `int8` for CPU |
| `LANGUAGE` | `pt` | Fixed to avoid misdetection from intro music/noise |
| `TIME_MARK_INTERVAL` | `300` | Seconds between time markers in the `.txt` |
| `PROGRESS_INTERVAL` | `300` | Seconds of audio between console progress lines |
| `CHUNK_LENGTH` | `30` | Audio chunk size in seconds (limits RAM usage) |

VAD filter (`vad_filter=True`, `min_silence_duration_ms=500`) suppresses background noise and music. Language is fixed to `pt` because instrumental intros can cause wrong language detection.
