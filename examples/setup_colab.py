import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    ("HeartMuLa/HeartMuLaGen", "ckpt"),
    (
        "HeartMuLa/HeartMuLa-oss-3B-happy-new-year",
        "ckpt/HeartMuLa-oss-3B",
    ),
    ("HeartMuLa/HeartCodec-oss-20260123", "ckpt/HeartCodec-oss"),
)


def install_dependencies():
    print("Instalando HeartMuLa y la interfaz Gradio...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".[webui]"],
        cwd=REPO_ROOT,
        check=True,
    )


def download_checkpoints():
    from huggingface_hub import snapshot_download

    for repo_id, local_dir in CHECKPOINTS:
        destination = REPO_ROOT / local_dir
        print(f"Descargando {repo_id} en {destination}...", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(destination))


def launch_webui(version: str):
    import torch

    if not torch.cuda.is_available():
        print(
            "AVISO: Colab no detecta una GPU. Activa una GPU en "
            "Entorno de ejecución > Cambiar tipo de entorno de ejecución.",
            flush=True,
        )
    else:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}", flush=True)

    print(
        "Iniciando la interfaz. Espera a que Gradio muestre la URL pública...",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "music_webui.py"),
            "--model_path",
            str(REPO_ROOT / "ckpt"),
            "--version",
            version,
            "--share",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Instala HeartMuLa, descarga los modelos y lanza Gradio para Colab."
    )
    parser.add_argument("--version", choices=["3B"], default="3B")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    install_dependencies()
    download_checkpoints()
    launch_webui(args.version)


if __name__ == "__main__":
    main()
