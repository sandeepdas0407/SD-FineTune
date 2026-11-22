"""Create the 4 local Ollama models from Q8_0 GGUFs in outputs/gguf/, with the same system prompts used in eval.

GGUFs come from llama.cpp's convert_hf_to_gguf.py (--outtype q8_0) run on outputs/merged/{base,uc1,uc2}.
(Ollama 0.40.x imports safetensors only via its MLX engine, which doesn't support Qwen2.)

  sd-support-base  = Qwen2.5-1.5B-Instruct + UC1 system prompt
  sd-support-ft    = UC1 fine-tuned          + UC1 system prompt
  sd-banking-base  = Qwen2.5-1.5B-Instruct + UC2 system prompt
  sd-banking-ft    = UC2 fine-tuned          + UC2 system prompt

Usage: python src/make_ollama.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import SYSTEM_UC1, system_uc2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MF_DIR = ROOT / "ollama"
OLLAMA = shutil.which("ollama") or str(Path.home() / "AppData/Local/Programs/Ollama/ollama.exe")

MODELS = {
    "sd-support-base": ("base", SYSTEM_UC1),
    "sd-support-ft": ("uc1", SYSTEM_UC1),
    "sd-banking-base": ("base", system_uc2()),
    "sd-banking-ft": ("uc2", system_uc2()),
}


# Qwen2.5 ChatML; the Modelfile SYSTEM is injected as the first message when a request has none
TEMPLATE = """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""


def modelfile(weights: str, system: str) -> str:
    assert '"""' not in system
    src = (ROOT / "outputs" / "gguf" / f"sd-{weights}-q8_0.gguf").as_posix()
    return (f"FROM {src}\n"
            f'TEMPLATE """{TEMPLATE}"""\n'
            f'SYSTEM """{system}"""\n'
            "PARAMETER temperature 0\n"      # greedy, same as the eval
            "PARAMETER num_ctx 4096\n"
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER stop "<|endoftext|>"\n')


def main():
    MF_DIR.mkdir(exist_ok=True)
    for name, (weights, system) in MODELS.items():
        mf = MF_DIR / f"{name}.Modelfile"
        mf.write_text(modelfile(weights, system), encoding="utf-8")
        print(f"== ollama create {name} from sd-{weights}-q8_0.gguf")
        subprocess.run([OLLAMA, "create", name, "-f", str(mf)], check=True)
    subprocess.run([OLLAMA, "list"], check=True)


if __name__ == "__main__":
    main()
