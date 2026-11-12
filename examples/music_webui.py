import argparse
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from heartlib import HeartMuLaGenPipeline


class MusicGenerator:
    def __init__(self, model_path: str, version: str):
        self.model_path = model_path
        self.version = version
        self.pipeline = None
        self.lock = threading.Lock()

    def _load_pipeline(self):
        if self.pipeline is None:
            if not Path(self.model_path).is_dir():
                raise gr.Error(
                    f"No se encontró la carpeta del modelo: {self.model_path}. "
                    "Descarga los checkpoints y vuelve a iniciar la app."
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mula_dtype = torch.float32
            if device.type == "cuda":
                mula_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )

            self.pipeline = HeartMuLaGenPipeline.from_pretrained(
                self.model_path,
                device=device,
                dtype={"mula": mula_dtype, "codec": torch.float32},
                version=self.version,
                lazy_load=True,
            )
        return self.pipeline

    def generate(
        self,
        lyrics: str,
        tags: str,
        duration_seconds: int,
        temperature: float,
        top_k: int,
        cfg_scale: float,
    ):
        if not lyrics.strip():
            raise gr.Error("Escribe la letra antes de generar.")
        if not tags.strip():
            raise gr.Error("Añade al menos una etiqueta musical.")

        with self.lock:
            pipeline = self._load_pipeline()
            output = tempfile.NamedTemporaryFile(
                prefix="heartmula-", suffix=".wav", delete=False
            )
            output_path = output.name
            output.close()
            try:
                with torch.no_grad():
                    pipeline(
                        {"lyrics": lyrics, "tags": tags},
                        max_audio_length_ms=duration_seconds * 1000,
                        save_path=output_path,
                        temperature=temperature,
                        topk=top_k,
                        cfg_scale=cfg_scale,
                    )
            except Exception:
                Path(output_path).unlink(missing_ok=True)
                raise

        return output_path, f"Listo · {datetime.now().strftime('%H:%M:%S')}"


def build_demo(generator: MusicGenerator):
    css = """
    .gradio-container { max-width: 1120px !important; }
    #hero { text-align: center; padding: 1.25rem 0 0.5rem; }
    #hero h1 { font-size: 2.6rem; margin-bottom: 0.3rem; }
    #hero p { opacity: 0.78; font-size: 1.05rem; }
    #generate { min-height: 3.2rem; font-size: 1.1rem; }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="violet", secondary_hue="pink", neutral_hue="slate"
        ),
        css=css,
        title="HeartMuLa · Estudio musical",
    ) as demo:
        gr.Markdown(
            """
            <div id="hero">
              <h1>🎼 HeartMuLa Studio</h1>
              <p>Convierte tus letras e ideas en una canción original.</p>
            </div>
            """,
            elem_id="hero",
        )

        with gr.Row():
            with gr.Column(scale=6):
                lyrics = gr.Textbox(
                    label="Letra",
                    placeholder="[Verse]\nEscribe aquí tu letra...\n\n[Chorus]\nTu estribillo",
                    lines=14,
                    max_lines=24,
                )
                tags = gr.Textbox(
                    label="Estilo y etiquetas",
                    placeholder="indie-pop, dreamy, female-vocals, guitar",
                    info="Separa las etiquetas con comas, sin necesidad de espacios.",
                )
                with gr.Accordion("Ajustes de generación", open=False):
                    duration = gr.Slider(
                        minimum=10,
                        maximum=240,
                        value=60,
                        step=5,
                        label="Duración máxima (segundos)",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Creatividad (temperature)",
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=100, value=50, step=1, label="Top-k"
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        label="Fuerza de las indicaciones (CFG)",
                    )
                generate_button = gr.Button(
                    "✨ Generar canción", variant="primary", elem_id="generate"
                )
                status = gr.Markdown("Los modelos se cargarán al generar por primera vez.")

            with gr.Column(scale=5):
                audio = gr.Audio(label="Tu canción", type="filepath")
                gr.Markdown(
                    "💡 **Consejo:** usa etiquetas concretas para describir género, "
                    "instrumentos, ambiente y tipo de voz."
                )

        generate_button.click(
            fn=generator.generate,
            inputs=[lyrics, tags, duration, temperature, top_k, cfg_scale],
            outputs=[audio, status],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description="HeartMuLa music generation Web UI")
    parser.add_argument("--model_path", default="./ckpt", help="Checkpoint root folder")
    parser.add_argument("--version", choices=["3B", "7B"], default="3B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link (useful for Google Colab)",
    )
    args = parser.parse_args()

    generator = MusicGenerator(os.path.expanduser(args.model_path), args.version)
    demo = build_demo(generator)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
