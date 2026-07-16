"""
Configuration centrale du pipeline de fine-tuning Whisper pour le fulfulde (fuv).

Toutes les valeurs sont modifiables ici sans toucher au reste du code.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # Dossier racine sur ton Google Drive contenant :
    #   - un sous-dossier "audio/" avec 0001.wav, 0002.wav, ...
    #   - un fichier metadata.json (voir format dans le README)
    drive_root: str = "/content/drive/MyDrive/fuv-stt-dataset"
    audio_subdir: str = "audio"
    metadata_filename: str = "metadata.json"

    # Proportion des donnees reservee a la validation (le reste = train)
    eval_split_ratio: float = 0.1

    # Frequence d'echantillonnage attendue par Whisper
    sampling_rate: int = 16000

    # Seed pour que le split train/val soit reproductible
    seed: int = 42


@dataclass
class ModelConfig:
    # Checkpoint de base. "openai/whisper-small" = bon compromis
    # qualite/vitesse pour un dataset de quelques heures sur GPU Colab (T4).
    base_model: str = "openai/whisper-small"

    # Langue "bootstrap" utilisee uniquement pour initialiser le token de
    # langue du tokenizer (le fulfulde n'existe pas nativement dans Whisper).
    # Le fine-tuning corrige la correspondance reelle : ce choix n'a
    # d'impact que sur l'initialisation, pas sur le resultat final.
    # On choisit une langue africaine deja geree par Whisper pour rester
    # dans une distribution phonetique proche.
    bootstrap_language: str = "Swahili"
    task: str = "transcribe"


@dataclass
class TrainConfig:
    output_dir: str = "/content/drive/MyDrive/fuv-stt-dataset/whisper-small-fuv"

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2  # batch effectif = 16

    learning_rate: float = 1e-5
    warmup_steps: int = 50
    max_steps: int = 1000  # a ajuster selon la taille reelle du dataset

    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    logging_steps: int = 25

    fp16: bool = True  # desactive automatiquement si pas de GPU compatible
    generation_max_length: int = 225
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False

    push_to_hub: bool = False  # passe a True + renseigne hub_model_id si tu veux publier
    hub_model_id: str = "bonopassale/whisper-small-fuv"


data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()
