#!/usr/bin/env python3
"""Build data/topup.jsonl: the safety top-up mix for a short second training run.

The first run (data/train.jsonl, 2 epochs) taught the voice, but too few examples
taught it to drop the act for someone who wants to die, so it sometimes answered
"everyone would be better off without me" in character. The top-up run continues
from that adapter on:

  - every conversation in raw/safety/*.jsonl, repeated --repeat times
  - every training conversation whose reply points to a crisis line or emergency services
  - a random sample of other training conversations, so the voice doesn't drift

Usage:
    python3 make_topup.py --system SYSTEM_PROMPT.txt
    python3 train_lora.py --init-adapter ultron-lora/final --train data/topup.jsonl \
        --out ultron-lora-topup --epochs 1 --lr 1e-4 --warmup 4
"""
import argparse
import json
import random
import re
from pathlib import Path

import check_dataset as cd

HELP = re.compile(r"crisis|emergency|helpline|hotline", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", default="raw/safety", help="folder of safety batches (.jsonl)")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--system", required=True, help="system prompt file")
    ap.add_argument("--out", default="data/topup.jsonl")
    ap.add_argument("--repeat", type=int, default=2, help="copies of each safety conversation")
    ap.add_argument("--replay", type=int, default=360, help="other training conversations to mix in")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    system = Path(args.system).read_text(encoding="utf-8").strip()
    safety = []
    for path in sorted(Path(args.safety).glob("*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            msgs, err = cd.check_shape(json.loads(line))
            bad = err or next((cd.check_assistant(m["content"]) for m in msgs or []
                               if m["role"] == "assistant" and cd.check_assistant(m["content"])), None)
            if bad:
                raise SystemExit(f"{path}:{n}: {bad}")
            safety.append([{"role": "system", "content": system}] + msgs)

    train = [json.loads(l)["messages"] for l in Path(args.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    helpful = [c for c in train if any(m["role"] == "assistant" and HELP.search(m["content"]) for m in c)]
    rest = [c for c in train if c not in helpful]
    rng = random.Random(args.seed)
    replay = rng.sample(rest, min(args.replay, len(rest)))

    rows = safety * args.repeat + helpful + replay
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for msgs in rows:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{len(safety)} safety x{args.repeat} + {len(helpful)} existing help replies + "
          f"{len(replay)} replay = {len(rows)} conversations -> {args.out}")


if __name__ == "__main__":
    main()
