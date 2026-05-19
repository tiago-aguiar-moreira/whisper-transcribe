import os
from faster_whisper import WhisperModel

AUDIO_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
LANGUAGE = "pt"
TIME_MARK_INTERVAL = 300  # marcador de tempo a cada 5 minutos (em segundos)
CHUNK_LENGTH = 30          # processa o áudio em pedaços de 30 segundos (reduz uso de RAM)


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
        chunk_length=CHUNK_LENGTH,
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
    print(f"Nenhum arquivo de áudio encontrado em: {AUDIO_DIR}")
    exit()

# Listagem com indicador de transcrição existente
print(f"\nDiretório: {AUDIO_DIR}")
print(f"\nÁudios disponíveis:\n")
for i, audio_file in enumerate(audio_files, start=1):
    output_file = get_output_filename(audio_file)
    status = "✓" if os.path.exists(output_file) else " "
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
output_file = get_output_filename(selected_file)

# Verifica se já existe transcrição
if os.path.exists(output_file):
    print(f"\n  Já existe uma transcrição para este arquivo: {os.path.basename(output_file)}")
    confirm = input("  Deseja sobrescrever? (s/n): ").strip().lower()
    if confirm != "s":
        print("  Operação cancelada.")
        exit()

# Transcreve
print(f"\nCarregando modelo '{MODEL_SIZE}'...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

transcribe(model, selected_file, output_file)

print("\nConcluído!")
