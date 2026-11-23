#!/usr/bin/env python3
"""LoRA fine-tune for the Ultron dataset, fast enough to run on a CPU.

Usage:
    python3 train_lora.py --model Qwen/Qwen3-4B-Instruct-2507 --out ultron-lora

Every training example starts with the same system prompt. Instead of
running that prompt once per conversation, each micro-batch runs it once
and shares its keys and values with all the conversations in the batch
(gradients still flow through it). With a 174-token system prompt and
~60-token conversations, that cuts the work per epoch by about 2.5x.

Loss is computed only on Ultron's replies (and their end-of-turn token).
--verify checks the shared-prefix loss and gradients against an ordinary
full-sequence forward pass before training.

Works on CUDA too (it uses the GPU when one is present), but on a GPU
you'd normally use Unsloth or TRL instead.
"""

import argparse
import gc
import json
import math
import os
import random
import time

# PyTorch's CPU matmul backend (oneDNN via ideep) caches a compiled kernel for
# every input shape. Variable-length batches make thousands of shapes, which
# grows memory by ~100 MB per step. Keep the caches small. (Set before torch loads.)
os.environ.setdefault("LRU_CACHE_CAPACITY", "64")
os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "64")

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:  # return freed heap memory to the OS between steps (glibc only)
    import ctypes
    _trim = ctypes.CDLL("libc.so.6").malloc_trim
except (OSError, AttributeError):
    _trim = None


def release_memory():
    gc.collect()
    if _trim:
        _trim(0)


def rss_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(msg, path=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def load_examples(path, tok, check=True):
    """Return (prefix_ids, [(suffix_ids, suffix_labels), ...])."""
    rows = [json.loads(l)["messages"] for l in open(path, encoding="utf-8") if l.strip()]
    system = rows[0][0]["content"]
    assert all(r[0]["role"] == "system" and r[0]["content"] == system for r in rows), \
        "every example must start with the same system prompt"
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    prefix = enc(f"<|im_start|>system\n{system}<|im_end|>\n")
    end, nl = enc("<|im_end|>"), enc("\n")
    out = []
    for msgs in rows:
        ids, labels = [], []
        for m in msgs[1:]:
            head = enc(f"<|im_start|>{m['role']}\n")
            body = enc(m["content"])
            ids += head + body + end + nl
            learn = m["role"] == "assistant"
            labels += [-100] * len(head) + (body + end if learn else [-100] * (len(body) + 1)) + [-100]
        if check:
            full = enc(tok.apply_chat_template(msgs, tokenize=False))
            assert prefix + ids == full, (
                "piecewise tokenization differs from the chat template; this script expects a "
                "non-thinking ChatML model such as Qwen3-4B-Instruct-2507")
        out.append((ids, labels))
    return prefix, out


def batches(examples, micro, seed, max_tokens=0):
    """Length-bucketed micro-batches of example indices, shuffled.

    Each holds at most `micro` conversations and, if max_tokens is set, at
    most that many padded tokens after the shared prefix (memory cap).
    """
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    chunk = micro * 32
    groups = []
    for i in range(0, len(idx), chunk):
        part = sorted(idx[i:i + chunk], key=lambda j: len(examples[j][0]))
        cur = []
        for j in part:  # ascending length, so j is the widest in cur
            width = len(examples[j][0])
            if cur and (len(cur) == micro or (max_tokens and width * (len(cur) + 1) > max_tokens)):
                groups.append(cur)
                cur = []
            cur.append(j)
        if cur:
            groups.append(cur)
    rng.shuffle(groups)
    return groups


def steps_from(groups, per_step):
    """Group micro-batches into optimizer steps of about per_step conversations."""
    steps, cur, n = [], [], 0
    for g in groups:
        if cur and n + len(g) > per_step:
            steps.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += len(g)
    if cur:
        steps.append(cur)
    return steps


def collate(examples, group, pad_id, device):
    width = max(len(examples[j][0]) for j in group)
    ids = torch.full((len(group), width), pad_id, dtype=torch.long)
    labels = torch.full((len(group), width), -100, dtype=torch.long)
    mask = torch.zeros((len(group), width), dtype=torch.long)
    for r, j in enumerate(group):
        s, l = examples[j]
        ids[r, :len(s)] = torch.tensor(s)
        labels[r, :len(l)] = torch.tensor(l)
        mask[r, :len(s)] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def recompute_in_backward(model):
    """Don't store MLP and RMSNorm activations; recompute them during backward.

    These blocks hold most of the activation memory (the MLP's 9728-wide
    intermediates, and RMSNorm's float32 copies) but don't touch the KV
    cache, so checkpointing them is safe with the shared prefix. Costs
    roughly 25% more compute and cuts activation memory by more than half.
    """
    decoder, _ = core(model)
    def wrap(mod):
        fwd = mod.forward
        def run(*a, **k):
            if torch.is_grad_enabled():
                return checkpoint(fwd, *a, use_reentrant=False, **k)
            return fwd(*a, **k)
        mod.forward = run
    for layer in decoder.layers:
        for mod in (layer.mlp, layer.input_layernorm, layer.post_attention_layernorm,
                    layer.self_attn.q_norm, layer.self_attn.k_norm):
            wrap(mod)


def core(model):
    """(decoder, lm_head) of a PEFT-wrapped or plain causal LM."""
    m = model.get_base_model() if isinstance(model, PeftModel) else model
    return m.model, m.lm_head


def shared_prefix_loss(model, prefix, ids, labels, mask):
    """Summed loss over learnable tokens, with the prefix computed once."""
    decoder, head = core(model)
    b = ids.shape[0]
    p = torch.tensor([prefix], device=ids.device)
    pre = decoder(input_ids=p, use_cache=True)
    layers = pre.past_key_values.layers
    cache = DynamicCache(ddp_cache_data=[
        (L.keys.expand(b, -1, -1, -1), L.values.expand(b, -1, -1, -1)) for L in layers
    ])
    full_mask = torch.cat([torch.ones((b, len(prefix)), dtype=mask.dtype, device=mask.device), mask], dim=1)
    hid = decoder(input_ids=ids, attention_mask=full_mask, past_key_values=cache, use_cache=True).last_hidden_state
    # the hidden state before each suffix token predicts it; the first suffix
    # token is <|im_start|>, never learned, so the prefix's last state is unused
    target = labels[:, 1:]
    keep = target != -100
    logits = head(hid[:, :-1][keep]).float()
    loss = F.cross_entropy(logits, target[keep], reduction="sum")
    loss._caches = (pre.past_key_values, cache)  # cleared by drop_caches() after backward
    return loss, int(keep.sum())


def drop_caches(loss):
    """Release the KV tensors held by a loss's caches (autograd keeps what it needs)."""
    for c in getattr(loss, "_caches", ()):
        for layer in getattr(c, "layers", []):
            layer.keys = layer.values = None


def full_sequence_loss(model, prefix, ids, labels, mask):
    """Reference implementation: ordinary forward over prefix + suffix."""
    decoder, head = core(model)
    b = ids.shape[0]
    p = torch.tensor([prefix] * b, device=ids.device)
    x = torch.cat([p, ids], dim=1)
    m = torch.cat([torch.ones_like(p), mask], dim=1)
    hid = decoder(input_ids=x, attention_mask=m).last_hidden_state[:, len(prefix):]
    target = labels[:, 1:]
    keep = target != -100
    logits = head(hid[:, :-1][keep]).float()
    return F.cross_entropy(logits, target[keep], reduction="sum"), int(keep.sum())


@torch.no_grad()
def evaluate(model, prefix, examples, micro, pad_id, device, max_tokens=0):
    model.eval()
    total, n = 0.0, 0
    for g in batches(examples, micro, seed=0, max_tokens=max_tokens):
        ids, labels, mask = collate(examples, g, pad_id, device)
        loss, k = shared_prefix_loss(model, prefix, ids, labels, mask)
        total, n = total + loss.item(), n + k
        drop_caches(loss)
    model.train()
    release_memory()
    return total / n


def verify(model, prefix, examples, pad_id, device, logf):
    g = list(range(3))
    ids, labels, mask = collate(examples, g, pad_id, device)
    params = [p for p in model.parameters() if p.requires_grad]
    with torch.no_grad():  # nonzero LoRA B so gradients reach lora_A too
        for p in params:
            p.add_(torch.randn_like(p) * 1e-3)
    out = []
    for fn in (full_sequence_loss, shared_prefix_loss):
        model.zero_grad()
        loss, k = fn(model, prefix, ids, labels, mask)
        (loss / k).backward()
        grads = torch.cat([p.grad.flatten().float() for p in params])
        out.append(((loss / k).item(), grads))
    (l1, g1), (l2, g2) = out
    cos = F.cosine_similarity(g1, g2, dim=0).item()
    rel = ((g1 - g2).norm() / g1.norm()).item()
    log(f"verify: full loss {l1:.5f} shared loss {l2:.5f} | grad cosine {cos:.6f} rel diff {rel:.4f}", logf)
    with torch.no_grad():
        for p in params:
            p.zero_()
    model.zero_grad()
    assert abs(l1 - l2) < 0.02 * abs(l1) and cos > 0.99, "shared-prefix pass does not match"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="ultron-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8, help="conversations per forward pass")
    ap.add_argument("--accum", type=int, default=2, help="micro-batches per optimizer step")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="cap on padded tokens per micro-batch after the prefix (0 = no cap)")
    ap.add_argument("--pass-tokens", type=int, default=0,
                    help="compute micro-batches larger than this in two halves (lower peak memory; "
                         "same math, same batches)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-recompute", action="store_true", help="store all activations (faster, more memory)")
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (for timing runs)")
    ap.add_argument("--restart-above-gb", type=float, default=0,
                    help="if memory between steps exceeds this, save a checkpoint and exit with code 3 "
                         "so a supervisor can resume in a fresh process")
    ap.add_argument("--verify", action="store_true", help="check the shared-prefix math, then exit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fp32", action="store_true", help="load the model in float32 (for --verify on small models)")
    ap.add_argument("--skip-template-check", action="store_true", help="for --verify on models with other templates")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    logf = os.path.join(args.out, "train.log")

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    check = not args.skip_template_check
    prefix, train = load_examples(args.train, tok, check)
    _, val = load_examples(args.val, tok, check)
    log(f"{len(train)} train / {len(val)} val conversations, shared prefix {len(prefix)} tokens", logf)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32 if args.fp32 else torch.bfloat16).to(device)
    model.config.use_cache = False
    if device == "cpu":
        # transformers maps the weights from the safetensors files lazily; under
        # memory pressure Linux drops those pages and re-reads them from disk on
        # every step. Copy them into ordinary memory once.
        with torch.no_grad():
            for p in model.parameters():
                p.data = p.data.clone()
    ckpt = os.path.join(args.out, "checkpoint")
    resume = os.path.exists(os.path.join(ckpt, "state.pt")) and not args.verify
    if resume:
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True, autocast_adapter_dtype=False)
    else:
        cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                         target_modules=TARGETS, task_type="CAUSAL_LM")
        # adapters run in the model's dtype (bf16): PEFT's default fp32 adapters
        # store an fp32 copy of every layer input, doubling activation memory
        model = get_peft_model(model, cfg, autocast_adapter_dtype=False)
    model.train()
    if not args.no_recompute:
        recompute_in_backward(model)
    params = [p for p in model.parameters() if p.requires_grad]
    # fp32 master copies for the optimizer, so small late updates aren't lost to bf16 rounding
    master = [p.detach().float().clone().requires_grad_(True) for p in params]
    log(f"trainable parameters: {sum(p.numel() for p in params):,} ({params[0].dtype}, fp32 master)", logf)

    if args.verify:
        verify(model, prefix, train, pad_id, device, logf)
        return

    # one flat list of optimizer steps over all epochs, so resuming is exact
    plan = []
    for e in range(math.ceil(args.epochs)):
        groups = batches(train, args.micro, args.seed + e, args.max_tokens)
        plan += steps_from(groups, args.micro * args.accum)
    steps_per_epoch = len(plan) / math.ceil(args.epochs)
    total_steps = min(len(plan), int(round(steps_per_epoch * args.epochs)))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW(master, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    step = 0
    if resume:
        state = torch.load(os.path.join(ckpt, "state.pt"), weights_only=False)
        with torch.no_grad():
            for m, saved in zip(master, state["master"]):
                m.copy_(saved)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step = state["step"]
        log(f"resumed from step {step}", logf)
    else:
        log(f"val loss before training: {evaluate(model, prefix, val, args.micro, pad_id, device, args.max_tokens):.4f}", logf)

    t0, done = time.time(), 0
    while step < total_steps:
        groups = plan[step]
        coll = [collate(train, g, pad_id, device) for g in groups]
        n_tok = sum(int((c[1][:, 1:] != -100).sum()) for c in coll)
        total = 0.0
        for ids, labels, mask in coll:
            parts = [(ids, labels, mask)]
            if args.pass_tokens and ids.numel() > args.pass_tokens and ids.shape[0] > 1:
                h = ids.shape[0] // 2
                parts = [(ids[:h], labels[:h], mask[:h]), (ids[h:], labels[h:], mask[h:])]
            for pi, pl, pm in parts:
                loss, _ = shared_prefix_loss(model, prefix, pi, pl, pm)
                (loss / n_tok).backward()
                total += loss.item()
                drop_caches(loss)
                del loss
        for m, p in zip(master, params):
            m.grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for m, p in zip(master, params):
                p.copy_(m)
        release_memory()
        step += 1
        done += 1
        rate = (time.time() - t0) / done
        n_conv = sum(len(g) for g in groups)
        log(f"step {step}/{total_steps} loss {total / n_tok:.4f} lr {sched.get_last_lr()[0]:.2e} "
            f"({n_conv} convs, {len(groups)} batches) {rate:.1f}s/step eta {(total_steps - step) * rate / 60:.0f}min "
            f"rss {rss_gb():.1f}GB", logf)
        if step % args.save_every == 0 or step == total_steps:
            model.save_pretrained(ckpt)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "master": [m.detach() for m in master]},
                       os.path.join(ckpt, "state.pt"))
        if args.restart_above_gb and step < total_steps and rss_gb() > args.restart_above_gb:
            model.save_pretrained(ckpt)
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "master": [m.detach() for m in master]}, os.path.join(ckpt, "state.pt"))
            log(f"memory {rss_gb():.1f}GB above {args.restart_above_gb}GB: checkpointed at step {step}, "
                f"exiting for a fresh process", logf)
            raise SystemExit(3)
        if step % args.eval_every == 0 or step == total_steps:
            log(f"val loss at step {step}: {evaluate(model, prefix, val, args.micro, pad_id, device, args.max_tokens):.4f}", logf)

    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    log(f"saved adapter to {final}", logf)


if __name__ == "__main__":
    main()
