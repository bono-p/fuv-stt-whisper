"""
Data collator specifique a Whisper : les features audio et les labels textuels
n'ont pas la meme logique de padding, donc on les traite separement puis
on recombine en un seul batch.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 1. Les features audio (log-mel) sont deja de taille fixe -> simple stack via padding
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 2. Les labels textuels doivent etre padded separement (longueur variable)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplace le padding par -100 pour que la loss l'ignore
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Retire le token BOS s'il a ete ajoute automatiquement par un
        # entrainement precedent (evite un decalage de sequence)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
