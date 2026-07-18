#!/usr/bin/env python3
"""
convert_audio.py
════════════════════════════════════════════════════════
Conversion des fichiers audio .m4a → .wav 16kHz mono
+ génération du metadata.json template pour Whisper STT

Projet Tardigrade — DevLab / M. Passale
────────────────────────────────────────────────────────
Usage :
  python convert_audio.py <input_dir> <output_dir> [options]

Exemples :
  # Conversion simple (garde les noms d'origine)
  python convert_audio.py ./raw_audio ./audio

  # Renommage séquentiel 0001.wav, 0002.wav, ...
  python convert_audio.py ./raw_audio ./audio --rename

  # Conversion + génération du metadata.json template
  python convert_audio.py ./raw_audio ./audio --rename --metadata

  # Changer le sample rate (défaut : 16000 Hz)
  python convert_audio.py ./raw_audio ./audio --rename --metadata --sr 22050
════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple


# ── Couleurs terminal ──────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def ok(msg):  print(f"{C.GREEN}✅ {msg}{C.RESET}")
def warn(msg): print(f"{C.YELLOW}⚠️  {msg}{C.RESET}")
def err(msg):  print(f"{C.RED}❌ {msg}{C.RESET}")
def info(msg): print(f"{C.CYAN}   {msg}{C.RESET}")


# ── Vérification ffmpeg ────────────────────────────────────────

def check_ffmpeg() -> bool:
    """Vérifie que ffmpeg est accessible dans le PATH."""
    if shutil.which("ffmpeg") is None:
        err("ffmpeg introuvable dans le PATH.")
        info("Windows  → https://ffmpeg.org/download.html  (ajouter ffmpeg/bin au PATH)")
        info("macOS    → brew install ffmpeg")
        info("Ubuntu   → sudo apt install ffmpeg")
        return False

    result = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True
    )
    version_line = result.stdout.split("\n")[0] if result.stdout else "version inconnue"
    ok(f"ffmpeg trouvé : {version_line}")
    return True


# ── Conversion d'un seul fichier ───────────────────────────────

def convert_file(
    src: Path,
    dst: Path,
    sample_rate: int = 16_000
) -> Tuple[bool, str]:
    """
    Convertit src (.m4a / .mp3 / .ogg / ...) vers dst (.wav).

    Paramètres ffmpeg :
      -ar <sr>      : sample rate cible
      -ac 1         : mono (1 canal)
      -c:a pcm_s16le: codec WAV non compressé 16-bit little-endian
      -y            : écraser le fichier de destination sans confirmation
      -loglevel error : silencieux sauf erreur

    Retourne (succès: bool, message: str).
    """
    cmd = [
        "ffmpeg",
        "-i",        str(src),
        "-ar",       str(sample_rate),
        "-ac",       "1",
        "-c:a",      "pcm_s16le",
        "-y",
        "-loglevel", "error",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        detail = result.stderr.strip().split("\n")[-1] if result.stderr else "erreur inconnue"
        return False, detail

    # Vérifier que le fichier a bien été créé et n'est pas vide
    if not dst.exists() or dst.stat().st_size == 0:
        return False, "fichier de sortie vide ou absent"

    return True, "ok"


# ── Barre de progression simple ────────────────────────────────

def progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total
    return f"[{bar}] {current:>{len(str(total))}}/{total}  {pct:5.1f}%"


# ── Génération du metadata.json ────────────────────────────────

def write_metadata(output_dir: Path, entries: List[dict]) -> Path:
    """
    Écrit le fichier metadata.json dans le dossier parent de output_dir
    (ou dans output_dir si le parent n'est pas accessible).

    Structure :
    [
      {"file": "0001.wav", "transcription": ""},
      ...
    ]

    Le champ 'transcription' est vide — à remplir manuellement.
    """
    metadata_path = output_dir.parent / "metadata.json"

    payload = [
        {"file": e["wav_name"], "transcription": ""}
        for e in entries
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return metadata_path


# ── Rapport final ──────────────────────────────────────────────

def print_report(
    entries: List[dict],
    failed: List[Tuple[str, str]],
    output_dir: Path,
    sample_rate: int,
    metadata_path: Path = None,
):
    total   = len(entries) + len(failed)
    n_ok    = len(entries)
    n_fail  = len(failed)

    print()
    print(f"{C.BOLD}{'═'*52}{C.RESET}")
    print(f"{C.BOLD}  RAPPORT DE CONVERSION{C.RESET}")
    print(f"{'─'*52}")
    print(f"  Fichiers traités   : {total}")
    print(f"  {C.GREEN}Convertis avec succès : {n_ok}{C.RESET}")
    if n_fail:
        print(f"  {C.RED}Échecs               : {n_fail}{C.RESET}")
        for fname, reason in failed:
            print(f"    • {fname} → {reason}")
    print(f"  Format de sortie   : WAV · {sample_rate} Hz · Mono · PCM 16-bit")
    print(f"  Dossier de sortie  : {output_dir}")

    if entries:
        # Durée totale estimée (taille WAV / octets par seconde)
        bps = sample_rate * 2  # 16-bit = 2 octets
        total_bytes = sum(
            (output_dir / e["wav_name"]).stat().st_size
            for e in entries
            if (output_dir / e["wav_name"]).exists()
        )
        total_sec  = total_bytes / bps
        total_min  = total_sec / 60
        print(f"  Durée totale estimée : {total_min:.1f} min ({total_sec:.0f}s)")

    if metadata_path:
        print(f"\n  {C.CYAN}metadata.json généré :{C.RESET} {metadata_path}")
        print(f"  {C.YELLOW}→ Remplissez le champ 'transcription' pour chaque fichier{C.RESET}")
        print(f"    avant de pousser sur Google Drive.")

    print(f"{C.BOLD}{'═'*52}{C.RESET}")


# ── Point d'entrée ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Conversion .m4a → .wav 16kHz pour fine-tuning Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir",
        help="Dossier source contenant les fichiers .m4a"
    )
    parser.add_argument(
        "output_dir",
        help="Dossier de sortie pour les fichiers .wav"
    )
    parser.add_argument(
        "--rename", "-r",
        action="store_true",
        help="Renommer en 0001.wav, 0002.wav, ... (sinon garde le nom d'origine)"
    )
    parser.add_argument(
        "--metadata", "-m",
        action="store_true",
        help="Générer un metadata.json template (transcriptions vides)"
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16_000,
        dest="sample_rate",
        help="Fréquence d'échantillonnage cible en Hz (défaut : 16000)"
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".m4a",
        dest="extension",
        help="Extension source à convertir (défaut : .m4a). Ex: --ext .mp3"
    )
    args = parser.parse_args()

    # ── Validation ───────────────────────────────────────────────

    if not check_ffmpeg():
        sys.exit(1)

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        err(f"Dossier source introuvable : {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    ext = args.extension.lower()
    if not ext.startswith("."):
        ext = "." + ext

    # ── Lister les fichiers sources ──────────────────────────────

    src_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() == ext
    )

    if not src_files:
        warn(f"Aucun fichier '{ext}' trouvé dans : {input_dir}")
        sys.exit(1)

    print()
    print(f"{C.BOLD}🎵 Conversion audio — Projet Tardigrade{C.RESET}")
    print(f"{'─'*52}")
    info(f"Source    : {input_dir}")
    info(f"Sortie    : {output_dir}")
    info(f"Fichiers  : {len(src_files)} fichier(s) '{ext}'")
    info(f"Format    : {args.sample_rate} Hz · Mono · WAV 16-bit")
    label_renommage = "oui (0001.wav, ...)" if args.rename else "non (nom d'origine)"
    info(f"Renommage : {label_renommage}")
    print()

    # ── Conversion ────────────────────────────────────────────────

    success_entries: List[dict] = []
    failed_entries:  List[Tuple[str, str]] = []

    for i, src in enumerate(src_files):
        # Nom de sortie
        if args.rename:
            wav_name = f"{i + 1:04d}.wav"
        else:
            wav_name = src.stem + ".wav"

        dst = output_dir / wav_name

        # Affichage progression
        bar = progress_bar(i, len(src_files))
        print(f"\r  {bar}  {src.name[:30]:<30}", end="", flush=True)

        ok_flag, message = convert_file(src, dst, args.sample_rate)

        if ok_flag:
            success_entries.append({
                "src_name": src.name,
                "wav_name": wav_name,
            })
        else:
            failed_entries.append((src.name, message))

    # Barre finale à 100%
    bar = progress_bar(len(src_files), len(src_files))
    print(f"\r  {bar}  {'':30}", flush=True)

    # ── Génération metadata.json ─────────────────────────────────

    metadata_path = None
    if args.metadata and success_entries:
        metadata_path = write_metadata(output_dir, success_entries)

    # ── Rapport ──────────────────────────────────────────────────

    print_report(
        success_entries,
        failed_entries,
        output_dir,
        args.sample_rate,
        metadata_path,
    )

    sys.exit(0 if not failed_entries else 1)


if __name__ == "__main__":
    main()
