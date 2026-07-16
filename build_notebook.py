import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Fine-tuning Whisper (small) pour le fulfulde (fuv) — Adamaoua, Cameroun

Ce notebook :
1. Monte ton Google Drive
2. Clone/utilise le code du repo `fuv-stt-whisper`
3. Installe les dependances
4. Verifie ton dataset (audio + `metadata.json`)
5. Lance le fine-tuning de `openai/whisper-small`
6. Teste le modele obtenu sur un nouvel audio

Avant de commencer : pousse ton dossier de dataset sur ton Drive avec cette structure :

```
MyDrive/fuv-stt-dataset/
├── audio/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── metadata.json
```

Voir le README du repo pour le format exact de `metadata.json`."""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Monter Google Drive"))
cells.append(nbf.v4.new_code_cell(
"""from google.colab import drive
drive.mount('/content/drive')"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 2. Récupérer le code du projet\\n\\n"
"Remplace l'URL par celle de ton repo GitHub une fois que tu l'auras créé et poussé."
))
cells.append(nbf.v4.new_code_cell(
"""%cd /content
!git clone https://github.com/bonopassale/fuv-stt-whisper.git
%cd fuv-stt-whisper"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Installer les dépendances"))
cells.append(nbf.v4.new_code_cell(
"""!pip install -r requirements.txt --quiet"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 4. Vérifier / ajuster la configuration\\n\\n"
"Ouvre `src/config.py` si besoin (chemin Drive, taille du modele, hyperparametres). "
"Par defaut tout pointe vers `/content/drive/MyDrive/fuv-stt-dataset`."
))
cells.append(nbf.v4.new_code_cell(
"""!cat src/config.py"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 5. Vérification rapide du dataset (avant d'entraîner)\\n\\n"
"Charge juste les métadonnées et vérifie que tous les fichiers audio existent, "
"sans lancer l'entraînement complet."
))
cells.append(nbf.v4.new_code_cell(
"""%cd src
from dataset import load_raw_dataset
raw = load_raw_dataset()
print(raw)
print("\\nExemple :", raw["train"][0]["text"])"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 6. Lancer le fine-tuning\\n\\n"
"Selon la taille de ton dataset (~2h d'audio) et le GPU Colab attribué, "
"compte entre 30 minutes et 2 heures pour `max_steps=1000`."
))
cells.append(nbf.v4.new_code_cell(
"""!python train.py"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 7. Tester le modèle fine-tuné\\n\\n"
"Dépose un fichier audio de test (fulfulde, différent du dataset d'entraînement) "
"et transcris-le."
))
cells.append(nbf.v4.new_code_cell(
"""!python inference.py --audio /content/drive/MyDrive/fuv-stt-dataset/test_audio.wav"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 8. (Optionnel) Publier sur Hugging Face Hub\\n\\n"
"Passe `push_to_hub = True` dans `config.py` avant l'entraînement, "
"ou pousse manuellement le dossier `output_dir` a posteriori :"
))
cells.append(nbf.v4.new_code_cell(
"""from huggingface_hub import notebook_login
notebook_login()

# Puis, si push_to_hub=False pendant l'entrainement :
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# from config import train_config
# model = WhisperForConditionalGeneration.from_pretrained(train_config.output_dir)
# processor = WhisperProcessor.from_pretrained(train_config.output_dir)
# model.push_to_hub("bonopassale/whisper-small-fuv")
# processor.push_to_hub("bonopassale/whisper-small-fuv")"""
))

nb['cells'] = cells

with open('/home/claude/fuv-stt-whisper/notebook/finetune_whisper_fuv.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook cree avec succes.")
