"""
Fine-tuning de Whisper (small par defaut) sur le dataset fulfulde (fuv).

Usage (depuis le notebook Colab, apres avoir monte Drive) :

    %cd /content/fuv-stt-whisper/src
    !python train.py

Le meilleur checkpoint (selon le WER de validation) est sauvegarde dans
train_config.output_dir, directement sur ton Google Drive pour survivre
au redemarrage de la VM Colab.
"""

import evaluate
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from config import data_config, model_config, train_config
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from dataset import build_processed_dataset

wer_metric = evaluate.load("wer")


def load_processor_and_model():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_config.base_model)

    # NOTE : le fulfulde n'existe pas dans la liste des langues Whisper.
    # On utilise une langue "bootstrap" proche (cf config.py) uniquement
    # pour initialiser les tokens speciaux <|lang|> et <|task|>. Le
    # fine-tuning reapprend la vraie correspondance texte <-> audio,
    # independamment de ce choix.
    tokenizer = WhisperTokenizer.from_pretrained(
        model_config.base_model,
        language=model_config.bootstrap_language,
        task=model_config.task,
    )

    processor = WhisperProcessor.from_pretrained(
        model_config.base_model,
        language=model_config.bootstrap_language,
        task=model_config.task,
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_config.base_model)

    # Desactive la contrainte de langue forcee : on laisse le modele
    # generer librement, ce qui est important puisqu'on sort du cadre
    # des langues originales de Whisper.
    model.generation_config.language = None
    model.generation_config.task = model_config.task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return feature_extractor, tokenizer, processor, model


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Remplace -100 par le pad token pour pouvoir decoder correctement
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def main():
    print("=== 1/4 : Chargement du modele et du processor ===")
    feature_extractor, tokenizer, processor, model = load_processor_and_model()

    print("\n=== 2/4 : Chargement et preparation du dataset ===")
    dataset = build_processed_dataset(feature_extractor, tokenizer, num_proc=1)

    print("\n=== 3/4 : Configuration de l'entrainement ===")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    use_fp16 = train_config.fp16 and torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        eval_strategy=train_config.eval_strategy,
        eval_steps=train_config.eval_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        logging_steps=train_config.logging_steps,
        fp16=use_fp16,
        predict_with_generate=True,
        generation_max_length=train_config.generation_max_length,
        load_best_model_at_end=train_config.load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        push_to_hub=train_config.push_to_hub,
        hub_model_id=train_config.hub_model_id if train_config.push_to_hub else None,
        report_to=["none"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=processor.feature_extractor,
    )

    print("\n=== 4/4 : Entrainement ===")
    trainer.train()

    print("\nSauvegarde du modele final...")
    trainer.save_model(train_config.output_dir)
    processor.save_pretrained(train_config.output_dir)

    if train_config.push_to_hub:
        trainer.push_to_hub()

    print(f"\n[OK] Modele sauvegarde dans : {train_config.output_dir}")


if __name__ == "__main__":
    main()
