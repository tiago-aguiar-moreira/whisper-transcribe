"""Transcreve um áudio do diretório atual, escolhido numa lista numerada.

Uso (a partir do diretório dos áudios):
    python transcribe.py            # .txt + .srt + .words.json
    python transcribe.py --srt      # .txt + .srt
    python transcribe.py --words    # .txt + .words.json

Cada execução grava um log em logs\\AAAA-MM-DD_HH-MM-SS.log (na pasta do repositório).
Códigos de saída: 0 ok, 2 falha, 130 interrompido (Ctrl+C).
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata

AUDIO_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MODEL_SIZE = "turbo"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANGUAGE = "pt"
TIME_MARK_INTERVAL = 300  # marcador de tempo no .txt a cada 5 minutos (em segundos)
PROGRESS_INTERVAL = 300   # progresso no console a cada 5 minutos de áudio (em segundos)
CHUNK_LENGTH = 30         # processa o áudio em pedaços de 30 segundos (reduz uso de RAM)
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".mkv", ".webm")
JSON_FORMAT_VERSION = 1
PACKAGES = ("faster-whisper", "ctranslate2", "av")
EXIT_OK, EXIT_FAILURES, EXIT_FATAL, EXIT_INTERRUPTED = 0, 1, 2, 130


# ---------------------------------------------------------------- argumentos e arquivos

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Transcreve os áudios do diretório atual com faster-whisper.",
        epilog="O .txt é sempre gerado. Sem --srt nem --words, gera os dois.")
    parser.add_argument("--srt", action="store_true", help="gera a legenda <nome>-transcript.srt")
    parser.add_argument("--words", action="store_true",
                        help="gera <nome>-transcript.words.json (tempo de cada palavra)")
    args = parser.parse_args(argv)
    if not (args.srt or args.words):
        args.srt = args.words = True
    return args


def find_audio_files():
    return sorted(os.path.join(AUDIO_DIR, f) for f in os.listdir(AUDIO_DIR) if f.endswith(AUDIO_EXTENSIONS))


def get_output_files(audio_file, srt=True, words=True):
    """Arquivos de saída por tipo. O sufixo "-new" só é removido do fim do nome:
    20260516-1-new.m4a → 20260516-1-transcript.txt / .srt / .words.json"""
    base = os.path.splitext(os.path.basename(audio_file))[0]
    if base.endswith("-new"):
        base = base[:-len("-new")]
    base = os.path.join(OUTPUT_DIR, f"{base}-transcript")
    files = {"txt": base + ".txt"}
    if srt:
        files["srt"] = base + ".srt"
    if words:
        files["words"] = base + ".words.json"
    return files


def is_done(files):
    return all(os.path.exists(path) for path in files.values())


# ---------------------------------------------------------------- formatos de saída

def build_transcript(segments, *, duration, language, language_probability, source):
    """Monta o resultado a partir dos segmentos do Whisper. Palavras ficam em ordem, sem
    sobreposição e dentro da duração do áudio (o Whisper às vezes estica o fim além dela)."""
    def clamp(value):
        return round(min(max(value, 0.0), duration), 3)

    result = []
    previous_end = 0.0
    for segment in segments:
        words = []
        for w in segment.words or []:
            text = w.word.strip()
            if not text:
                continue
            if words and not w.word[:1].isspace():
                # Continuação da palavra anterior ("bem" + "-vindo", "100" + "%"): palavra nova
                # começa com espaço no resultado bruto do Whisper.
                last = words[-1]
                last["word"] += text
                last["end"] = previous_end = clamp(max(w.end, last["end"]))
                last["probability"] = min(last["probability"], round(w.probability, 3))
                continue
            start = clamp(max(w.start, previous_end))
            end = clamp(max(w.end, start))
            previous_end = end
            words.append({"start": start, "end": end, "word": text, "probability": round(w.probability, 3)})
        result.append({"start": clamp(segment.start), "end": clamp(segment.end),
                       "text": segment.text.strip(), "words": words})
    return {"version": JSON_FORMAT_VERSION, "language": language,
            "language_probability": round(language_probability, 4), "source": source,
            "duration": round(duration, 3), "segments": result}


def format_txt(data):
    lines = []
    last_mark = -TIME_MARK_INTERVAL
    for segment in data["segments"]:
        if segment["start"] - last_mark >= TIME_MARK_INTERVAL:
            lines.append(f"\n[{int(segment['start'] // 60)}min]")
            last_mark = segment["start"]
        lines.append(segment["text"])
    header = f"Idioma: {data['language']} ({data['language_probability']:.2%})\n\n"
    return header + " ".join(lines)


def _srt_time(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt(data):
    return "\n".join(f"{i}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{s['text']}\n"
                     for i, s in enumerate(data["segments"], start=1))


def format_words_json(data):
    """JSON indentado com uma palavra por linha, para dar para ler e corrigir num editor."""
    def one(value):
        return json.dumps(value, ensure_ascii=False)

    header = ",\n".join(f"  {one(k)}: {one(v)}" for k, v in data.items() if k != "segments")
    segments = []
    for s in data["segments"]:
        fields = ",\n".join(f"      {one(k)}: {one(v)}" for k, v in s.items() if k != "words")
        words = ",\n".join(f"        {one(w)}" for w in s["words"])
        words = f"[\n{words}\n      ]" if words else "[]"
        segments.append(f"    {{\n{fields},\n      \"words\": {words}\n    }}")
    body = "[\n" + ",\n".join(segments) + "\n  ]" if segments else "[]"
    return f"{{\n{header},\n  \"segments\": {body}\n}}\n"


FORMATTERS = {"srt": format_srt, "words": format_words_json, "txt": format_txt}


def write_outputs(contents):
    """Grava cada arquivo num .tmp e só depois troca pelos definitivos, com o .txt por último
    (o .txt existir implica os outros gravados). Falha ou Ctrl+C apagam os .tmp."""
    tmps = [path + ".tmp" for path, _ in contents]
    try:
        for tmp, (_, text) in zip(tmps, contents):
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
        for tmp, (path, _) in zip(tmps, contents):
            os.replace(tmp, path)
    except BaseException:
        for tmp in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise


# ---------------------------------------------------------------- transcrição

def load_model():
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def _minutes(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}min{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _with_progress(segments, total_seconds):
    next_report = 0
    for segment in segments:
        if segment.start >= next_report:
            print(f"    ... {int(segment.start // 60)}min de {int(total_seconds // 60)}min", flush=True)
            next_report = segment.start + PROGRESS_INTERVAL
        yield segment


def transcribe(model, audio_file, files):
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        chunk_length=CHUNK_LENGTH,
        word_timestamps="words" in files,
    )
    print(f"  Idioma: {info.language} ({info.language_probability:.2%}) | áudio: {_minutes(info.duration)}",
          flush=True)
    data = build_transcript(_with_progress(segments, info.duration), duration=info.duration,
                            language=info.language, language_probability=info.language_probability,
                            source=os.path.basename(audio_file))
    write_outputs([(files[kind], FORMATTERS[kind](data)) for kind in ("srt", "words", "txt") if kind in files])
    for path in files.values():
        print(f"  Salvo em: {path}", flush=True)


# ---------------------------------------------------------------- log da execução

class _Tee:
    """Escreve no console e no log ao mesmo tempo."""

    def __init__(self, console, run_log):
        self._console = console
        self._log = run_log

    def write(self, text):
        self._console.write(text)
        self._log.feed(text)
        return len(text)

    def flush(self):
        self._console.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._console, name)


class RunLog:
    """Log por execução: espelha o console com horário em cada linha; detail() só vai para o arquivo."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8", newline="\n")
        self._buffer = ""

    def tee(self, console):
        return _Tee(console, self)

    def _write_line(self, line):
        if line.strip():
            self._file.write(f"{datetime.now():%H:%M:%S} {line.rstrip()}\n")

    def feed(self, text):
        self._buffer += text
        *lines, self._buffer = re.split(r"[\r\n]", self._buffer)
        for line in lines:
            self._write_line(line)

    def detail(self, text):
        for line in text.splitlines():
            self._file.write(f"{datetime.now():%H:%M:%S}     | {line}\n")
        self._file.flush()

    def flush(self):
        self._file.flush()

    def close(self):
        self._write_line(self._buffer)
        self._buffer = ""
        self._file.close()


def _versions():
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name} (ausente)")
    return f"Python {sys.version.split()[0]} | " + " | ".join(found)


def print_header(started, args):
    outputs = ", ".join(ext for ext, on in ((".txt", True), (".srt", args.srt), (".words.json", args.words)) if on)
    print("=" * 60)
    print(f"Início    : {started:%d/%m/%Y %H:%M:%S}")
    print(f"Diretório : {AUDIO_DIR}")
    print(f"Saídas    : {outputs}")
    print(f"Config    : modelo={MODEL_SIZE} device={DEVICE} compute={COMPUTE_TYPE} idioma={LANGUAGE}")
    print(f"Versões   : {_versions()}")
    print("=" * 60, flush=True)


# ---------------------------------------------------------------- interativo

def main(argv=None):
    # Console do Windows / saída redirecionada usam cp1252: sem isso, "✓" e acentos quebram o print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    audio_files = find_audio_files()
    if not audio_files:
        print(f"Nenhum arquivo de áudio encontrado em: {AUDIO_DIR}")
        return EXIT_OK

    # Listagem com indicador de transcrição existente (todas as saídas pedidas)
    print(f"\nDiretório: {AUDIO_DIR}")
    print("\nÁudios disponíveis:\n")
    for i, audio_file in enumerate(audio_files, start=1):
        status = "✓" if is_done(get_output_files(audio_file, args.srt, args.words)) else " "
        print(f"  [{i}] [{status}] {os.path.basename(audio_file)}")
    print("\n  [✓] = já transcrito\n")

    # Seleção do usuário
    while True:
        try:
            choice = int(input("Digite o número do arquivo que deseja transcrever: "))
            if 1 <= choice <= len(audio_files):
                break
            print(f"  Número inválido. Digite entre 1 e {len(audio_files)}.")
        except ValueError:
            print("  Entrada inválida. Digite apenas o número.")

    selected_file = audio_files[choice - 1]
    files = get_output_files(selected_file, args.srt, args.words)

    # Verifica se já existe alguma das saídas
    existing = [os.path.basename(p) for p in files.values() if os.path.exists(p)]
    if existing:
        print(f"\n  Já existe transcrição para este arquivo: {', '.join(existing)}")
        confirm = input("  Deseja sobrescrever? (s/n): ").strip().lower()
        if confirm != "s":
            print("  Operação cancelada.")
            return EXIT_OK

    started = datetime.now()
    run_log = RunLog(os.path.join(LOG_DIR, f"{started:%Y-%m-%d_%H-%M-%S}.log"))
    original = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = run_log.tee(sys.stdout), run_log.tee(sys.stderr)
    code = EXIT_FATAL
    try:
        print()
        print_header(started, args)
        try:
            print(f"\nCarregando modelo '{MODEL_SIZE}'...", flush=True)
            model = load_model()
            print(f"\nTranscrevendo: {os.path.basename(selected_file)}", flush=True)
            transcribe(model, selected_file, files)
            code = EXIT_OK
        except KeyboardInterrupt:
            print("\nINTERROMPIDO", flush=True)
            code = EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001 — registra o traceback no log e sai com código 2
            print(f"\nFALHA: {exc}", flush=True)
            run_log.detail(traceback.format_exc())

        end = datetime.now()
        print("\nConcluído!" if code == EXIT_OK else "\nEncerrado sem concluir.")
        print("-" * 60)
        print(f"  Início    : {started:%d/%m/%Y %H:%M:%S}")
        print(f"  Fim       : {end:%d/%m/%Y %H:%M:%S}")
        print(f"  Duração   : {str(end - started).split('.')[0]}")
        print(f"  Log       : {run_log.path}")
        print("-" * 60)
        print(f"Fim (código {code})", flush=True)
    finally:
        sys.stdout, sys.stderr = original
        run_log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
