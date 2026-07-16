"""
Chargement du dataset fulfulde (audio + texte) depuis Google Drive et
preparation pour l'entrainement Whisper (features log-mel + labels tokenises).

Format attendu de metadata.json (a la racine de data_config.drive_root) :

[
  {"file_name": "0001.wav", "text": "texte fulfulde correspondant"},
  {"file_name": "0002.wav", "text": "autre phrase en fulfulde", "speaker": "voix_1"},
  ...
]

- "file_name" : nom du fichier dans audio_subdir (pas le chemin complet)
- "text"      : transcription exacte de l'audio
- "speaker"   : optionnel, juste garde comme metadonnee, non utilise pour l'instant

Le champ "speaker" est optionnel et sert seulement a garder une tracabilite
si tu melanges plusieurs voix (utile plus tard pour analyser le WER par voix).
"""

import json
import os

from datasets import Audio, Dataset, DatasetDict

from config import data_config


def _load_metadata(metadata_path: str) -> list:
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            "metadata.json doit contenir une LISTE d'objets "
            '[{"file_name": ..., "text": ...}, ...]. '
            f"Type recu : {type(records)}"
        )
    return records


def _validate_and_build_paths(records: list, audio_dir: str) -> list:
    """Verifie que chaque fichier audio existe reellement, construit le chemin complet."""
    valid_records = []
    missing = []

    for rec in records:
        if "file_name" not in rec or "text" not in rec:
            continue
        full_path = os.path.join(audio_dir, rec["file_name"])
        if os.path.isfile(full_path):
            valid_records.append({"audio": full_path, "text": rec["text"].strip()})
        else:
            missing.append(rec["file_name"])

    if missing:
        print(f"[!] {len(missing)} fichier(s) audio introuvable(s), ignore(s) :")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... et {len(missing) - 10} autres")

    if not valid_records:
        raise RuntimeError(
            "Aucun enregistrement valide trouve. Verifie que audio_subdir "
            "et metadata_filename pointent bien vers les bons fichiers."
        )

    return valid_records


def load_raw_dataset() -> DatasetDict:
    """Charge metadata.json + audios, fait le split train/val, retourne un DatasetDict."""
    metadata_path = os.path.join(data_config.drive_root, data_config.metadata_filename)
    audio_dir = os.path.join(data_config.drive_root, data_config.audio_subdir)

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"metadata.json introuvable : {metadata_path}\n"
            "Verifie que ton Drive est bien monte et que le chemin dans "
            "config.py (data_config.drive_root) est correct."
        )

    records = _load_metadata(metadata_path)
    records = _validate_and_build_paths(records, audio_dir)

    print(f"[OK] {len(records)} paires audio/texte valides chargees.")

    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=data_config.sampling_rate))

    split = dataset.train_test_split(
        test_size=data_config.eval_split_ratio,
        seed=data_config.seed,
    )
    dataset_dict = DatasetDict({"train": split["train"], "eval": split["test"]})

    print(
        f"[OK] Split effectue -> train: {len(dataset_dict['train'])} | "
        f"eval: {len(dataset_dict['eval'])}"
    )
    return dataset_dict


def prepare_features(batch, feature_extractor, tokenizer):
    """Transforme un batch brut (audio + text) en features log-mel + labels tokenises.
    A utiliser avec dataset.map(..., remove_columns=...)."""
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def build_processed_dataset(feature_extractor, tokenizer, num_proc: int = 1) -> DatasetDict:
    """Pipeline complet : chargement brut + extraction des features + tokenisation."""
    raw = load_raw_dataset()

    processed = raw.map(
        lambda batch: prepare_features(batch, feature_extractor, tokenizer),
        remove_columns=raw["train"].column_names,
        num_proc=num_proc,
        desc="Extraction des features audio + tokenisation du texte",
    )
    return processed
