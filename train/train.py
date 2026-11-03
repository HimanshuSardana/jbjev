"""LoRA fine-tune. Run on Kaggle (T4, fp16). Early stop on valid MAE.

Kaggle setup (2 cells):
  cell 1: !pip install -q transformers>=5 peft accelerate scipy scikit-learn
  cell 2: upload data/ as a dataset or clone the repo, then:  !python train/train.py --data data --out out
"""
from __future__ import annotations
import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from . import config as C
from .data import load_rows, split_row, to_prompt
from .model import JevHead, JevModel, build_backbone, apply_lora


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        resume, job = split_row(r["text"])
        enc = self.tok(to_prompt(resume, job, C.TEMPLATE), truncation=True,
                       max_length=C.MAX_LEN)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "score": torch.tensor(r["score"], dtype=torch.float32),
                "verdict": torch.tensor(r["verdict"], dtype=torch.long)}


def collate(batch, tok):
    ids = [b["input_ids"] for b in batch]
    msk = [b["attention_mask"] for b in batch]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    L = max(len(x) for x in ids)
    inp = torch.tensor([x + [pad] * (L - len(x)) for x in ids])
    att = torch.tensor([x + [0] * (L - len(x)) for x in msk])
    return {"input_ids": inp, "attention_mask": att,
            "score": torch.stack([b["score"] for b in batch]),
            "verdict": torch.stack([b["verdict"] for b in batch])}


@torch.no_grad()
def eval_mae(model, loader, device) -> float:
    model.eval()
    se, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        s, _ = model(b["input_ids"], b["attention_mask"])
        se += F.mse_loss(s, b["score"], reduction="sum").item()
        n += len(s)
    return (se / n) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_rows = load_rows(f"{args.data}/train.jsonl")
    valid_rows = load_rows(f"{args.data}/valid.jsonl")
    cw = torch.tensor([1147 / 355, 1147 / 366, 1147 / 426], dtype=torch.float32)  # recompute if data changes
    cw = (cw / cw.sum() * 3).to(device)

    dl_train = DataLoader(PairDS(train_rows, tok), batch_size=C.BATCH, shuffle=True,
                          collate_fn=lambda b: collate(b, tok))
    dl_valid = DataLoader(PairDS(valid_rows, tok), batch_size=C.BATCH * 2,
                          collate_fn=lambda b: collate(b, tok))

    base = build_backbone(args.model)
    base = apply_lora(base, C.LORA_R, C.LORA_ALPHA, C.LORA_DROPOUT, C.LORA_TARGETS)
    base.print_trainable_parameters()
    model = JevModel(base, JevHead()).to(device).train()
    if device == "cuda":
        model = model.to(dtype=torch.float16) if False else model  # heads stay fp32; autocast handles it

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    total = len(dl_train) * args.epochs // C.GRAD_ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(total * C.WARMUP_RATIO), total)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best, bad, step = float("inf"), 0, 0
    import os
    os.makedirs(args.out, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        for i, b in enumerate(dl_train):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
                s, v = model(b["input_ids"], b["attention_mask"])
                loss = (C.LOSS_SCORE_W * F.mse_loss(s, b["score"])
                        + C.LOSS_VERDICT_W * F.cross_entropy(v, b["verdict"], weight=cw))
            loss = loss / C.GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % C.GRAD_ACCUM == 0:
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(); step += 1
        mae = eval_mae(model, dl_valid, device)
        print(f"epoch {ep}: valid MAE {mae:.4f} (best {best:.4f})", flush=True)
        if mae < best - 1e-4:
            best, bad = mae, 0
            torch.save({"lora": base.state_dict(), "head": model.head.state_dict(),
                        "tok": args.model, "mae": mae}, f"{args.out}/best.pt")
        else:
            bad += 1
            if bad >= C.PATIENCE:
                print("early stop"); break
    print("best valid MAE:", best)


if __name__ == "__main__":
    main()
