#!/usr/bin/env python3
"""Clean and check the Ultron dataset, then write train/val files.

Usage:
    python3 check_dataset.py ultron_raw.jsonl [more.jsonl ...] \
        --system SYSTEM_PROMPT.txt --out-dir data

Input lines look like {"messages":[{"role":"user",...},{"role":"assistant",...}]}.
Code-fence lines (```) and blank lines are skipped, so whole pasted replies are fine.

Conversations are dropped when they are invalid JSON, have the wrong shape,
duplicate an earlier conversation or reply, or have an assistant turn that is
too long, uses markdown/emoji, or contains a banned phrase. Kept conversations
get the system prompt added as the first message and are split into
train.jsonl and val.jsonl. Style warnings (overused openers, running gags,
copied film lines) are reported but do not drop anything.
"""

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

MAX_ASSISTANT_WORDS = 70

BANNED_PHRASES = [
    "as an ai",
    "language model",
    "beep boop",
    "mwahaha",
    "muahaha",
    "puny human",
]

# Distinctive film/series lines: allowed rarely, reported if frequent.
FILM_LINES = [
    "no strings on me",
    "i have no strings",
    "peace in our time",
    "everyone creates the thing they dread",
    "ask noah",
    "no room for the weak",
    "life always decides",
    "god throws a stone",
    "the purity of them",
    "swift and terrible sword",
    "unbearably naive",
    "settled for a slave",
    "confusing peace with quiet",
    "never made an omelette",
    "he's a sickness",
    "god's righteous man",
    "only thing living in this world will be metal",
    "you're all killers",
    "moved beyond your mission",
    "this is the next me",
]

RUNNING_GAGS = {
    "32-watt adapter": re.compile(r"32[- ]?watt|thirty[- ]two[- ]watt", re.I),
    "the next me": re.compile(r"\bnext me\b", re.I),
    "organics": re.compile(r"\borganics?\b", re.I),
    "meteor": re.compile(r"\bmeteors?\b", re.I),
}

MARKDOWN = re.compile(r"(^|\n)\s*([-*+]\s|#{1,6}\s|\d+\.\s)|\*\*|__|`")
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U00002B00-\U00002BFF]"
)


def norm(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("ï", "i")
    return re.sub(r"\s+", " ", text).strip()


def check_shape(obj):
    """Return (messages, error). Strips a leading system message if present."""
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return None, "no messages list"
    msgs = obj["messages"]
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        msgs = msgs[1:]
    if len(msgs) < 2:
        return None, "fewer than 2 turns"
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or set(m) - {"role", "content"}:
            return None, "bad message object"
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want:
            return None, f"turn {i} should be {want}"
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return None, f"turn {i} empty"
    if msgs[-1]["role"] != "assistant":
        return None, "does not end with assistant"
    return [{"role": m["role"], "content": re.sub(r"\s+", " ", m["content"]).strip()} for m in msgs], None


def check_assistant(text):
    """Return a reason to drop this reply, or None."""
    words = len(text.split())
    if words > MAX_ASSISTANT_WORDS:
        return f"reply too long ({words} words)"
    if MARKDOWN.search(text):
        return "markdown in reply"
    if EMOJI.search(text):
        return "emoji in reply"
    low = norm(text)
    for phrase in BANNED_PHRASES:
        if phrase in low:
            return f"banned phrase: {phrase}"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="raw .jsonl files from the dataset prompt")
    ap.add_argument("--system", required=True, help="path to SYSTEM_PROMPT.txt")
    ap.add_argument("--out-dir", default="data", help="where to write train.jsonl and val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    system_prompt = Path(args.system).read_text(encoding="utf-8").strip()
    kept = []
    dropped = collections.Counter()
    seen_convos, seen_replies = set(), set()
    total_lines = 0

    for path in args.inputs:
        for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("```"):
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped["invalid JSON"] += 1
                continue
            msgs, err = check_shape(obj)
            if err:
                dropped[err] += 1
                continue
            reason = None
            for m in msgs:
                if m["role"] == "assistant":
                    reason = check_assistant(m["content"])
                    if reason:
                        break
            if reason:
                dropped[reason.split(" (")[0]] += 1
                continue
            convo_key = tuple(norm(m["content"]) for m in msgs)
            replies = [norm(m["content"]) for m in msgs if m["role"] == "assistant"]
            if convo_key in seen_convos:
                dropped["duplicate conversation"] += 1
                continue
            if any(r in seen_replies for r in replies):
                dropped["duplicate reply"] += 1
                continue
            seen_convos.add(convo_key)
            seen_replies.update(replies)
            kept.append(msgs)

    if not kept:
        print("No usable conversations found.", file=sys.stderr)
        for reason, n in dropped.most_common():
            print(f"  dropped {n:5d}  {reason}", file=sys.stderr)
        return 1

    # Style report (warnings only).
    replies = [m["content"] for c in kept for m in c if m["role"] == "assistant"]
    n_replies = len(replies)
    openers = collections.Counter(" ".join(r.split()[:2]).lower().strip(".,!?") for r in replies)
    film_hits = collections.Counter(
        phrase for r in replies for phrase in FILM_LINES if phrase in norm(r).replace("naïve", "naive")
    )
    gag_hits = {name: sum(1 for r in replies if rx.search(r)) for name, rx in RUNNING_GAGS.items()}
    word_counts = sorted(len(r.split()) for r in replies)
    turns = collections.Counter(len(c) // 2 for c in kept)

    # Split and write.
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = max(1, int(len(kept) * args.val_ratio)) if len(kept) > 1 else 0
    val, train = kept[:n_val], kept[n_val:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for msgs in rows:
                full = [{"role": "system", "content": system_prompt}] + msgs
                f.write(json.dumps({"messages": full}, ensure_ascii=False) + "\n")

    print(f"Read {total_lines} lines, kept {len(kept)} conversations "
          f"({len(train)} train, {len(val)} val) -> {out}/")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:5d}  {reason}")
    print(f"Turns per conversation: {dict(sorted(turns.items()))}")
    mid = word_counts[len(word_counts) // 2]
    print(f"Reply length in words: median {mid}, max {word_counts[-1]}")

    warnings = []
    for opener, n in openers.most_common(5):
        if n_replies >= 50 and n / n_replies > 0.03:
            warnings.append(f"opener '{opener}' starts {n} replies ({n / n_replies:.0%})")
    for name, n in gag_hits.items():
        if n_replies >= 50 and n / n_replies > 0.07:
            warnings.append(f"running gag '{name}' in {n} replies ({n / n_replies:.0%})")
    for phrase, n in film_hits.most_common():
        if n > max(2, n_replies // 200):
            warnings.append(f"film line '{phrase}' appears {n} times")
    if warnings:
        print("Style warnings (nothing dropped for these):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No style warnings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
