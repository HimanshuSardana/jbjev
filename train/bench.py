"""CPU latency bench. Run on the target box before optimising anything.

    python train/bench.py --ckpt out/best.pt --data data --n 64

Reports, in order of what actually matters:
  1. token-length distribution of real rows  (if p99 is 180, max_len=512 is pure waste)
  2. fp32 latency at REAL lengths vs padded-to-512
  3. int8 dynamic-quant latency
  4. thread scaling, batch scaling
"""
from __future__ import annotations
import argparse, json, statistics, time
import torch
from transformers import AutoTokenizer

from . import config as C
from .data import load_rows, split_row, to_prompt


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def timeit(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), pct(ts, 0.95)


def load(ckpt_path, model_id, int8=False):
    from .model import JevHead, JevModel, build_backbone, apply_lora
    ck = torch.load(ckpt_path, map_location="cpu")
    mid = ck.get("tok", model_id)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = build_backbone(mid, gradient_checkpointing=False)
    base = apply_lora(base, C.LORA_R, C.LORA_ALPHA, C.LORA_DROPOUT, C.LORA_TARGETS)
    base.load_state_dict(ck["lora"], strict=False)
    head = JevHead(); head.load_state_dict(ck["head"])
    model = JevModel(base, head).eval()
    if int8:
        # dynamic int8 on every nn.Linear (attn q/k/v/o, conv in/out, MLP, our heads)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/best.pt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    rows = load_rows(f"{args.data}/test.jsonl")[: args.n]
    tok, model = load(args.ckpt, args.model, int8=args.int8)
    prompts = [to_prompt(*split_row(r["text"]), C.TEMPLATE) for r in rows]

    # 1. real token lengths
    lens = [len(tok(p, truncation=True, max_length=C.MAX_LEN)["input_ids"]) for p in prompts]
    print(f"tokens  n={len(lens)}  p50={pct(lens,.5)}  p90={pct(lens,.9)}  p99={pct(lens,.99)}  max={max(lens)}")
    print(f"max_len setting = {C.MAX_LEN}  → padding waste at p50 = {C.MAX_LEN/pct(lens,.5):.1f}x")

    def run(batch, pad_to=None):
        def f():
            for i in range(0, len(prompts), batch):
                chunk = prompts[i:i + batch]
                kw = dict(truncation=True, max_length=C.MAX_LEN, padding=True, return_tensors="pt")
                if pad_to:
                    kw["max_length"] = pad_to
                    kw["padding"] = "max_length"
                enc = tok(chunk, **kw)
                with torch.inference_mode():
                    model(enc["input_ids"], enc["attention_mask"])
        return f

    print("\nthreads  batch   pad      p50 ms   p95 ms   pairs/s")
    for th in (2, 4, 8):
        torch.set_num_threads(th)
        for batch in (1, 8, 32):
            for pad in (None, 512):
                med, p95 = timeit(run(batch, pad), warmup=1, reps=5)
                per_pair = med / min(batch, len(prompts))
                print(f"{th:7d}  {batch:5d}   {'dyn' if pad is None else str(pad):>5}   "
                      f"{med:7.1f}   {p95:7.1f}   {1000/ per_pair:7.1f}")

    print(f"\nint8={args.int8}  threads={torch.get_num_threads()}")


if __name__ == "__main__":
    main()
