import os
from faster_whisper import WhisperModel

AUDIO_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "pt"
TIME_MARK_INTERVAL = 300  # marcador de tempo a cada 5 minutos (em segundos)


def get_output_filename(audio_file):
    base = os.path.splitext(os.path.basename(audio_file))[0]  # ex: 20260516-1-new
    base = base.replace("-new", "")                            # ex: 20260516-1
    return os.path.join(OUTPUT_DIR, f"{base}-transcript.txt") # ex: 20260516-1-transcript.txt


def transcribe(model, audio_file, output_file):
    print(f"\nTranscrevendo: {os.path.basename(audio_file)}")
    print("-" * 60)

    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language=LANGUAGE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    print(f"Idioma: {info.language} ({info.language_probability:.2%})")
    print("-" * 60)

    lines = []
    last_mark = -TIME_MARK_INTERVAL

    for segment in segments:
        text = segment.text.strip()

        if segment.start - last_mark >= TIME_MARK_INTERVAL:
            minutes = int(segment.start // 60)
            marker = f"\n[{minutes}min]"
            lines.append(marker)
            print(marker)
            last_mark = segment.start

        lines.append(text)
        print(text)

    transcript = " ".join(lines)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Idioma: {info.language} ({info.language_probability:.2%})\n\n")
        f.write(transcript)

    print("-" * 60)
    print(f"Salvo em: {output_file}")


# Coleta todos os arquivos .m4a do diretório atual
audio_files = sorted([
    os.path.join(AUDIO_DIR, f)
    for f in os.listdir(AUDIO_DIR)
    if f.endswith(".m4a")
])

if not audio_files:
    print("Nenhum arquivo de áudio encontrado.")
    exit()

# Filtra os que já foram transcritos
pending = [f for f in audio_files if not os.path.exists(get_output_filename(f))]
skipped = len(audio_files) - len(pending)

print(f"Áudios encontrados : {len(audio_files)}")
print(f"Já transcritos     : {skipped}")
print(f"A transcrever      : {len(pending)}")

if not pending:
    print("\nTodos os áudios já foram transcritos.")
    exit()

print(f"\nCarregando modelo '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

for i, audio_file in enumerate(pending, start=1):
    output_file = get_output_filename(audio_file)
    print(f"\n[{i}/{len(pending)}] {os.path.basename(audio_file)}")
    transcribe(model, audio_file, output_file)

print("\nLote concluído!")
