# Desk Ultron: the trained model

This branch holds only the model. The code, dataset and instructions are on the main working branch.

| File | What it is |
|---|---|
| `parts/ultron-q4_k_m.gguf.part_000` … `part_026` | The model (Qwen3-4B-Instruct-2507 fine-tuned as Ultron, quantized to Q4_K_M), split into 27 pieces of 90 MiB because GitHub refuses files over 100 MB |
| `ultron-q4_k_m.gguf.sha256` | Checksum of the whole file |
| `adapter/` | The LoRA adapter on its own, for merging into the base model and quantizing differently (for example Q5_K_M or Q8_0) |

## Rebuild the file on the Pi

```bash
git clone --branch bigfile-transfer --single-branch --depth 1 https://github.com/mitarthpathak/Ultron ultron-model
```
```bash
cd ultron-model && cat parts/ultron-q4_k_m.gguf.part_* > ultron-q4_k_m.gguf
```
```bash
sha256sum -c ultron-q4_k_m.gguf.sha256
```

It should print `ultron-q4_k_m.gguf: OK`. Then `rm -r parts` frees 2.5 GB, and you can run it:

```bash
~/llama.cpp/build/bin/llama-server -m ultron-q4_k_m.gguf -c 2048 -t 4 --port 8080
```

Send `SYSTEM_PROMPT.txt` as the system message on every request, and run `safety_net.py` on what the person said before the model answers. The model alone points someone in crisis to help in about 72% of replies on unseen test messages. The keyword check is there for the rest.
