"""Side-by-side: base vs fine-tuned, via the local Ollama API (http://127.0.0.1:11434). Standard library only.

  python src/ollama_compare.py support                      # interactive, UC1 models
  python src/ollama_compare.py banking "my card got eaten by the ATM"
"""
import json
import shutil
import sys
import textwrap
import time
import urllib.request

PAIRS = {"support": ("sd-support-base", "sd-support-ft"), "banking": ("sd-banking-base", "sd-banking-ft")}
URL = "http://127.0.0.1:11434/api/chat"


def ask(model, text):
    body = json.dumps({"model": model, "stream": False, "messages": [{"role": "user", "content": text}]}).encode()
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600) as r:
        d = json.load(r)
    tok_s = d.get("eval_count", 0) / max(1e-9, d.get("eval_duration", 1) / 1e9)
    return d["message"]["content"].strip(), time.time() - t0, tok_s


def show(text, pair):
    width = max(40, (shutil.get_terminal_size((120, 20)).columns - 3) // 2)
    cols = []
    for m in pair:
        out, secs, tps = ask(m, text)
        lines = [f"[{m}]  {secs:.1f}s, {tps:.0f} tok/s", "-" * width]
        for para in out.splitlines() or [""]:
            lines += textwrap.wrap(para, width) or [""]
        cols.append(lines)
    for i in range(max(map(len, cols))):
        left = cols[0][i] if i < len(cols[0]) else ""
        right = cols[1][i] if i < len(cols[1]) else ""
        print(f"{left:<{width}} | {right}")
    print()


def main():
    uc = sys.argv[1] if len(sys.argv) > 1 else "support"
    pair = PAIRS[uc]
    if len(sys.argv) > 2:
        show(" ".join(sys.argv[2:]), pair)
        return
    print(f"Comparing {pair[0]} vs {pair[1]}. Type a customer message (empty line to quit).")
    while (text := input("> ").strip()):
        show(text, pair)


if __name__ == "__main__":
    main()
