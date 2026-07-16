"""
Transcrit un fichier audio avec le modele Whisper fine-tune sur le fulfulde.

Usage :
    python inference.py --audio /chemin/vers/fichier.wav
    python inference.py --audio /chemin/vers/fichier.wav --model_dir /autre/chemin
"""

import argparse

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import train_config


def load_audio(path: str, target_sr: int = 16000):
    speech, sr = torchaudio.load(path)
    if sr != target_sr:
        speech = torchaudio.functional.resample(speech, sr, target_sr)
    return speech.mean(dim=0).numpy()  # mono


def transcribe(audio_path: str, model_dir: str) -> str:
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    speech = load_audio(audio_path)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(device)

    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], max_new_tokens=225)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcription fulfulde avec Whisper fine-tune")
    parser.add_argument("--audio", required=True, help="Chemin vers le fichier audio a transcrire")
    parser.add_argument(
        "--model_dir",
        default=train_config.output_dir,
        help="Dossier du modele fine-tune (par defaut : celui defini dans config.py)",
    )
    args = parser.parse_args()

    text = transcribe(args.audio, args.model_dir)
    print(f"\nTranscription : {text}")


if __name__ == "__main__":
    main()
