"""Eval: test + real. MAE, band accuracy, apply F1, Spearman, boundary slice, calibration bins.

Usage:  python train/evaluate.py --data data --ckpt out/best.pt --split test
        python train/evaluate.py --data data --ckpt out/best.pt --split real
"""
from __future__ import annotations
import argparse, json
import torch
from transformers import AutoTokenizer

from . import config as C
from .data import load_rows, split_row, to_prompt
from .model import JevHead, JevModel, build_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="out/best.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    tok = AutoTokenizer.from_pretrained(ckpt.get("tok", C.MODEL_ID), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = build_backbone(ckpt.get("tok", C.MODEL_ID), gradient_checkpointing=False)
    from .model import apply_lora
    base = apply_lora(base, C.LORA_R, C.LORA_ALPHA, C.LORA_DROPOUT, C.LORA_TARGETS)
    base.load_state_dict(ckpt["lora"], strict=False)
    head = JevHead(); head.load_state_dict(ckpt["head"])
    model = JevModel(base, head).to(device).eval()

    rows = load_rows(f"{args.data}/{args.split}.jsonl")
    prompts = [to_prompt(*split_row(r["text"]), C.TEMPLATE) for r in rows]
    ps, pv, _ = __import__("train.model", fromlist=["predict"]).predict(
        prompts, tok, model, max_len=C.MAX_LEN, device=device)

    import statistics
    ae = [abs(p - r["score"]) for p, r in zip(ps, rows)]
    band = [ (0 if p < 0.40 else 1 if p < 0.70 else 2) for p in ps ]
    acc = sum(b == r["verdict"] for b, r in zip(band, rows)) / len(rows)
    ap_ = [1 if p >= 0.50 else 0 for p in ps]
    tp = sum(1 for p, r in zip(ap_, rows) if p == 1 and r["apply"] == 1)
    fp = sum(1 for p, r in zip(ap_, rows) if p == 1 and r["apply"] == 0)
    fn = sum(1 for p, r in zip(ap_, rows) if p == 0 and r["apply"] == 1)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    # Spearman without scipy
    def ranks(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs); j = 0
        while j < len(o):
            k = j
            while k + 1 < len(o) and xs[o[k + 1]] == xs[o[j]]: k += 1
            m = (j + k) / 2
            for t in range(j, k + 1): r[o[t]] = m
            j = k + 1
        return r
    rp, rt = ranks(ps), ranks([r["score"] for r in rows])
    mp, mt = statistics.fmean(rp), statistics.fmean(rt)
    cov = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
    sp = (cov / ((sum((a - mp) ** 2 for a in rp) * sum((b - mt) ** 2 for b in rt)) ** 0.5 + 1e-9))
    edge = [(p, r) for p, r in zip(ps, rows)
            if any(abs(r["score"] - t) <= 0.05 for t in (0.40, 0.50, 0.70))]
    eacc = (sum(1 for p, r in edge
                if (0 if p < 0.40 else 1 if p < 0.70 else 2) == r["verdict"]) / len(edge)) if edge else float("nan")
    print(json.dumps({
        "split": args.split, "n": len(rows),
        "mae": round(statistics.fmean(ae), 4),
        "band_acc": round(acc, 4), "apply_f1": round(f1, 4),
        "spearman": round(sp, 4),
        "boundary_n": len(edge), "boundary_acc": round(eacc, 4) if edge else None,
        "mean_pred": round(statistics.fmean(ps), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
