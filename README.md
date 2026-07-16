# fuv-stt-whisper

Fine-tuning de **Whisper (small)** pour la reconnaissance vocale (STT) en
**fulfulde, variété Adamaoua (Cameroun)** — code ISO 639-3 `fuv`.

Projet développé dans le cadre de l'écosystème NLLB fra↔fuv
(`bonopassale/nllb-fra-fuv-finetuned`), en réutilisant le même esprit de
pipeline reproductible et documenté.

---

## 1. Pourquoi Whisper et pas MMS ?

Le fulfulde n'a pas de token de langue natif ni dans Whisper ni dans les
adaptateurs MMS pré-entraînés. Whisper a été choisi ici parce que :

- son vocabulaire (BPE) est **partagé entre toutes les langues** : il peut
  encoder n'importe quel texte UTF-8, même dans une langue qu'il n'a jamais
  vue, juste avec un peu plus de tokens par mot au début. Le fine-tuning
  réapprend directement la correspondance audio → texte fulfulde.
- des fine-tunings de Whisper sur des langues africaines à faibles
  ressources (15-20h d'audio annoté) donnent des résultats exploitables,
  et notre volume cible (~2h, plusieurs voix) est un bon point de départ
  pour un premier prototype, quitte à l'enrichir ensuite.
- `whisper-small` (244M paramètres) tient confortablement sur un GPU T4
  gratuit de Colab, en fine-tuning complet ou en LoRA si besoin plus tard.

**Point technique important** : comme le fulfulde n'existe pas dans la
liste des langues Whisper, on utilise un code de langue "bootstrap"
(configurable dans `src/config.py`, `Swahili` par défaut) uniquement pour
initialiser proprement les tokens spéciaux `<|lang|>` et `<|task|>` du
tokenizer. Ce choix n'affecte **que l'initialisation** — le fine-tuning sur
tes données fulfulde réapprend la vraie correspondance texte/audio, quel
que soit le code choisi au départ.

---

## 2. Structure du projet

```
fuv-stt-whisper/
├── README.md
├── requirements.txt
├── build_notebook.py              # génère notebook/finetune_whisper_fuv.ipynb
├── notebook/
│   └── finetune_whisper_fuv.ipynb # notebook Colab prêt à l'emploi
└── src/
    ├── config.py           # tous les paramètres modifiables (chemins, hyperparams)
    ├── dataset.py           # chargement du dataset + préparation des features
    ├── data_collator.py     # padding dynamique des batches (spécifique Whisper)
    ├── train.py              # script d'entraînement (Seq2SeqTrainer)
    └── inference.py          # transcription d'un nouveau fichier audio
```

---

## 3. Format attendu de tes données

Sur ton Google Drive, organise ton dataset ainsi :

```
MyDrive/fuv-stt-dataset/
├── audio/
│   ├── 0001.wav
│   ├── 0002.wav
│   ├── 0003.wav
│   └── ...
└── metadata.json
```

`metadata.json` doit être une **liste d'objets JSON**, un par audio :

```json
[
  {"file_name": "0001.wav", "text": "texte fulfulde correspondant exactement à l'audio"},
  {"file_name": "0002.wav", "text": "autre phrase en fulfulde", "speaker": "voix_1"},
  {"file_name": "0003.wav", "text": "encore une phrase", "speaker": "voix_2"}
]
```

Règles importantes :

- `file_name` : juste le **nom du fichier** (pas le chemin complet), il doit
  exister dans `audio/`.
- `text` : la transcription **exacte** de ce qui est dit (pas de
  ponctuation fantaisiste, reste cohérent sur la casse/orthographe d'un
  bout à l'autre du corpus — plus c'est cohérent, mieux le modèle apprend).
- `speaker` : **optionnel**, juste informatif pour l'instant (utile plus
  tard si tu veux analyser le WER par voix). Le script l'ignore pour
  l'entraînement mais ne plante pas s'il est présent.
- Formats audio : `.wav` recommandé. D'autres formats lus par `torchaudio`
  fonctionnent aussi, mais `.wav` mono 16kHz évite toute conversion surprise.
- Le script `dataset.py` **vérifie automatiquement** que chaque fichier
  listé existe bien, et ignore proprement (avec un avertissement) les
  entrées dont l'audio est manquant — donc tu peux pousser ton JSON même
  si l'enregistrement de certains fichiers est encore en cours.

---

## 4. Utilisation (Google Colab)

1. Pousse ton dossier `fuv-stt-dataset/` (audio + `metadata.json`) sur ton
   Google Drive, à la racine de "Mon Drive" (ou ajuste `drive_root` dans
   `src/config.py` si tu préfères un autre emplacement).
2. Crée ton repo GitHub avec le contenu de ce dossier, puis ajuste l'URL
   `git clone` dans la cellule 2 du notebook.
3. Ouvre `notebook/finetune_whisper_fuv.ipynb` dans Google Colab.
4. **Runtime → Change runtime type → GPU (T4)**.
5. Exécute les cellules dans l'ordre :
   - Montage de Drive
   - Clone du repo + installation des dépendances
   - Vérification rapide du dataset (sans entraînement, pour repérer vite
     un souci de chemin ou de fichier manquant)
   - Lancement du fine-tuning
   - Test du modèle sur un audio de validation
   - (Optionnel) Publication sur le Hugging Face Hub

Le meilleur checkpoint (selon le WER de validation) est sauvegardé
directement sur ton Drive (`whisper-small-fuv/` par défaut), donc il
survit à un redémarrage de la VM Colab.

---

## 5. Ajuster les hyperparamètres

Tout se passe dans `src/config.py` :

| Paramètre | Rôle | Valeur par défaut |
|---|---|---|
| `eval_split_ratio` | % des données réservé à la validation | 0.1 |
| `base_model` | Taille du modèle Whisper | `whisper-small` |
| `bootstrap_language` | Langue "placeholder" pour le tokenizer (voir §1) | `Swahili` |
| `max_steps` | Nombre de steps d'entraînement | 1000 |
| `learning_rate` | Taux d'apprentissage | 1e-5 |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | Batch effectif | 8 × 2 = 16 |

**Avec ~2h d'audio (plusieurs voix)** : commence avec les valeurs par
défaut. Si tu vois le WER de validation stagner ou remonter avant
`max_steps`, réduis `max_steps` ou augmente `eval_steps`/`save_steps` pour
repérer le meilleur point plus tôt (`load_best_model_at_end=True` s'en
charge automatiquement).

---

## 6. Tester le modèle après entraînement

```bash
python src/inference.py --audio /chemin/vers/nouveau_fichier.wav
```

Par défaut le script charge le modèle depuis `train_config.output_dir`
(donc directement le résultat du fine-tuning). Tu peux pointer vers un
autre dossier avec `--model_dir`.

---

## 7. Limites connues et pistes d'amélioration

- **2h d'audio, c'est un bon prototype, pas un modèle de production.**
  Attends-toi à un WER correct sur du vocabulaire proche de ton corpus
  d'entraînement, mais fragile sur du vocabulaire ou des locuteurs très
  différents. Élargir le corpus (plus de voix, plus de contextes/thèmes)
  améliore directement la robustesse.
- **Cohérence orthographique** : comme le fulfulde n'a pas toujours une
  orthographe standardisée stricte selon les locuteurs, essaie de garder
  une convention cohérente dans `metadata.json` (même si elle diffère de
  la norme académique), sinon le modèle "hésite" entre plusieurs graphies
  pour le même son.
- **Prochaine étape naturelle** : une fois un premier modèle obtenu,
  générer des prédictions sur de l'audio non vu, corriger les erreurs
  manuellement, et réinjecter ces corrections dans `metadata.json` pour un
  second cycle de fine-tuning (active learning artisanal, très efficace en
  faibles ressources).
- Le projet **TTS avec ta propre voix** est volontairement laissé de côté
  ici — c'est un sujet distinct que tu as évoqué pour plus tard.
