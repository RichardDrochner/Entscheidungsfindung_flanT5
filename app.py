import os
import time
import csv
from pathlib import Path
from datetime import datetime

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import webbrowser
import threading

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "flan-t5-base"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "perf_log.csv"

def append_log(row: dict):
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def load_model():
    if not MODEL_DIR.exists():
        raise RuntimeError(
            f"Modellordner nicht gefunden: {MODEL_DIR}\n"
            "Bitte Modell vorher herunterladen (siehe README)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        local_files_only=True
    )

    model.to(device)
    model.eval()
    return tokenizer, model, device


TOKENIZER, MODEL, DEVICE = load_model()

@torch.inference_mode()
def generate(
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
):
    if not instruction or not instruction.strip():
        raise gr.Error("Bitte eine Eingabe/Instruktion eingeben.")

    # Seed (für Reproduzierbarkeit)
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    # Tokenize
    inputs = TOKENIZER(
        instruction,
        return_tensors="pt",
        truncation=True
    ).to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    # Bei temperature==0.0 -> deterministic (greedy). Sonst sampling.
    do_sample = temperature > 0.0

    out_ids = MODEL.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        num_beams=1,
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    text = TOKENIZER.decode(out_ids[0], skip_special_tokens=True)

    vram_peak_mb = None
    if DEVICE == "cuda":
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": int(seed),
        "time_ms": round((t1 - t0) * 1000, 2),
        "vram_peak_mb": round(vram_peak_mb, 1) if vram_peak_mb is not None else None,
        "prompt_len": len(instruction),
        "output_len": len(text),
    }
    append_log(row)

    return text

with gr.Blocks(title="FLAN-T5 (Text → Text)") as demo:
    gr.Markdown("# FLAN-T5-base (Text → Text)\nInstruktion eingeben → Generate.")

    instruction = gr.Textbox(
        label="Instruktion / Prompt",
        placeholder="z.B. Fasse folgenden Text in 3 Stichpunkten zusammen: ...",
        lines=6
    )

    with gr.Row():
        max_new_tokens = gr.Slider(16, 512, value=128, step=1, label="Max new tokens")
        temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature (0 = deterministisch)")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = zufällig)")

    btn = gr.Button("Generate")
    out = gr.Textbox(label="Antwort", lines=10)

    btn.click(generate, inputs=[instruction, max_new_tokens, temperature, top_p, seed], outputs=out)

def open_browser():
    webbrowser.open("http://127.0.0.1:7863")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    demo.launch(server_name="127.0.0.1", server_port=7863, share=False, inbrowser=False, prevent_thread_lock=False)