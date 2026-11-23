# Desk Ultron: a ~4B talking model for a Raspberry Pi 4 robot

The plan: Claude Opus 5.5 writes about 2,000 conversations in Ultron's voice, you fine-tune a ~4B open model on them using a free cloud GPU, then you run the result on the Pi with a microphone and a speaker.

| File | What it is |
|---|---|
| `ULTRON_MIND.md` | Character bible: his psychology (12 drives), the desk-robot version, voice rules, boundaries, verified quotes and sources |
| `OPUS_DATASET_PROMPT.md` | The prompt for Claude Opus 5.5 that writes the dataset in 50 batches |
| `SYSTEM_PROMPT.txt` | The short system prompt the robot runs with (also added to every training example) |
| `check_dataset.py` | Cleans Opus's output, drops bad lines, warns about repetition, writes `train.jsonl` / `val.jsonl` |

## Step 1: generate the dataset with Claude Opus 5.5

Follow the instructions at the top of `OPUS_DATASET_PROMPT.md`. Put every batch into one file, `ultron_raw.jsonl`. Pasting whole replies, code fences included, is fine.

- **In the Claude app:** 50 replies. Your plan's usage limits may spread this across a few sessions.
- **Through the API:** model `claude-opus-5-5` costs $4 per million input tokens and $20 per million output tokens. All 50 batches should cost roughly $5–$15 (an estimate), or about half that through the Message Batches API.

Then clean and split:

```bash
python3 check_dataset.py ultron_raw.jsonl --system SYSTEM_PROMPT.txt --out-dir data
```

Read the style warnings. If one opener, gag or film line shows up too often, regenerate those batches or delete some of those lines. A 4B model will repeat whatever is overused.

## Step 2: pick the base model

Recommended: **Qwen3-4B-Instruct-2507** (Apache 2.0 license, strong for its size, and no hidden "thinking" step to waste the Pi's few words per second). Alternatives: Gemma 3 4B, or Phi-4-mini-instruct (3.8B). If a newer 4B instruct model is out, use a non-thinking variant.

## Step 3: fine-tune on a free GPU

This session has no GPU, so train in Google Colab (free T4) or on Kaggle (about 30 free GPU hours a week):

- Method: QLoRA (4-bit base model plus a LoRA adapter) with Unsloth or Hugging Face TRL.
- Starting settings: LoRA rank 16, alpha 16–32, learning rate 2e-4, 2 epochs, max sequence length 1024, loss on assistant replies only.
- Use `data/val.jsonl` to make sure validation loss isn't rising (overfitting). Chat with the model before exporting.
- Export: merge the adapter, convert to GGUF, and quantize to **Q4_K_M** (llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize`, or Unsloth's GGUF export). Expect a file of about 2.5 GB.

## Step 4: run it on the Raspberry Pi 4

Use 64-bit Raspberry Pi OS. A 4 GB Pi is the minimum for a 4B model; 8 GB is more comfortable once speech recognition runs too. Add a heatsink and fan: under load the Pi throttles at about 80 °C.

```bash
sudo apt install -y git cmake build-essential
```
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
```
```bash
cmake -B build && cmake --build build --config Release -j4
```
```bash
./build/bin/llama-server -m ~/ultron-q4_k_m.gguf -c 2048 -t 4 --port 8080
```

That serves an OpenAI-compatible chat API at `http://localhost:8080/v1/chat/completions`. Send `SYSTEM_PROMPT.txt` as the system message on every request.

**What to expect for speed:** a 4B model on a Pi 4 is slow. It's limited by memory bandwidth, so plan for roughly 1–2 words per second. A two-sentence reply takes about 20–40 seconds. That's an estimate; measure on your own board. Ways to speed it up:
- Speak each sentence as soon as it's generated, instead of waiting for the full reply.
- Keep replies short; the dataset already teaches this.
- Train a 1.5–2B model (for example Qwen3-1.7B) on the same dataset as a faster backup.
- A Raspberry Pi 5 runs it roughly two to three times faster.

## Step 5: give it a voice and ears

- **Hearing:** whisper.cpp with the `tiny.en` or `base.en` model, or Vosk. Both run on a Pi 4.
- **Speaking:** Piper TTS, which is fast on a Pi. For a robotic edge, pass the audio through `sox` (for example `pitch -250` and `flanger`) and tune it by ear.
- Please don't clone James Spader's real voice. Pick a deep Piper voice and add effects instead.
- **The loop:** microphone → whisper.cpp → llama-server → Piper → sox → speaker.
- **Motors or servos:** if the robot gets them later, let the model request actions only from a small whitelist that ordinary code carries out. Never let the model drive hardware directly.

## Notes

- Ultron is a Marvel character. This is fine for a personal robot. If you ever publish or sell the model, rename it and make the character your own.
- The research behind the persona, with sources, is in `ULTRON_MIND.md`. VisionQuest is excluded on purpose.
