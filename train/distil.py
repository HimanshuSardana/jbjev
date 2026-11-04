"""DistilBERT student: full fine-tune on the rubric labels. CPU-serving target.

    python train/distil.py --data data --out out_distil
    python train/distil.py --data data --out out_distil --eval test
    python train/distil.py --data data --out out_distil --eval real

Why not distillation: labels are exact rubric output (2dp). The 1.7B teacher learned
that same rubric to MAE 0.09 — distilling would copy its error. Train on score directly.

Measured on the target box (8-core Kaby Lake, 4 threads):
    DistilBERT 66M @ 72 tok = 35 ms/pair, 270 GFLOPS, 270 MB RSS
    max_len=192 covers p99 (real rows: p50=72 p90=90 p99=131 max=131)
"""
from __future__ import annotations
import argparse, json, os, random, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

MODEL = "distilbert-base-uncased"
HIDDEN = 768
MAX_LEN = 192          # p99 = 131 tokens; 512 was 6.7x wasted compute
TEMPLATE = "Premise: {resume}\nHypothesis: this candidate matches the job: {job}\nAnswer: score"
BATCH, LR, WD, EPOCHS, PATIENCE, W_S, W_V, SEED = 32, 2e-5, 0.01, 8, 2, 0.7, 0.3, 4242


def split_row(t: str):
    a, b = t.split("\n[SEP]\n", 1) if "\n[SEP]\n" in t else t.split("[SEP]", 1)
    a = a.strip(); b = b.strip()
    if a.startswith("resume:"): a = a[7:].strip()
    if b.startswith("job:"): b = b[3:].strip()
    return a, b


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        a, b = split_row(r["text"])
        enc = self.tok(TEMPLATE.format(resume=a, job=b), truncation=True, max_length=MAX_LEN)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "score": torch.tensor(r["score"], dtype=torch.float32),
                "verdict": torch.tensor(r["verdict"], dtype=torch.long)}


class Student(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL)
        self.score = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.verdict = nn.Linear(hidden, 3)
    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0]                                   # [CLS]
        return torch.sigmoid(self.score(cls)).squeeze(-1), self.verdict(cls)


def load(f): return [json.loads(l) for l in open(f) if l.strip()]


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); ps, pv, ts, tv = [], [], [], []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.amp.autocast("cuda", enabled=(dev == "cuda"), dtype=torch.float16):
            s, v = model(b["input_ids"], b["attention_mask"])
        ps += s.float().cpu().tolist(); pv += v.argmax(-1).cpu().tolist()
        ts += b["score"].cpu().tolist(); tv += b["verdict"].cpu().tolist()
    return ps, pv, ts, tv


def metrics(ps, pv, ts, tv, thr=(0.40, 0.70, 0.50)):
    t1, t2, ta = thr
    band = [0 if p < t1 else 1 if p < t2 else 2 for p in ps]
    mae = statistics.fmean(abs(p - t) for p, t in zip(ps, ts))
    acc = sum(a == b for a, b in zip(band, tv)) / len(ps)
    ap = [1 if p >= ta else 0 for p in ps]
    tp = sum(1 for p, t in zip(ap, ts) if p and t >= ta)
    fp = sum(1 for p, t in zip(ap, ts) if p and t < ta)
    fn = sum(1 for p, t in zip(ap, ts) if not p and t >= ta)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    def rk(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); j = 0
        while j < len(o):
            k = j
            while k + 1 < len(o) and xs[o[k + 1]] == xs[o[j]]: k += 1
            m = (j + k) / 2
            for t in range(j, k + 1): r[o[t]] = m
            j = k + 1
        return r
    rp, rt = rk(ps), rk(ts)
    mp, mt = statistics.fmean(rp), statistics.fmean(rt)
    cov = sum((a - mp) * (b - mt) for a, b in zip(rp, rt))
    sp = cov / ((sum((a - mp) ** 2 for a in rp) * sum((b - mt) ** 2 for b in rt)) ** 0.5 + 1e-9)
    edge = [(p, t) for p, t in zip(ps, ts) if any(abs(t - x) <= 0.05 for x in (0.40, 0.50, 0.70))]
    eacc = sum(1 for p, t in edge if (0 if p < t1 else 1 if p < t2 else 2) ==
               (0 if t < 0.40 else 1 if t < 0.70 else 2)) / max(1, len(edge))
    return {"mae": round(mae, 4), "band_acc": round(acc, 4), "apply_f1": round(f1, 4),
            "spearman": round(sp, 4), "boundary_n": len(edge), "boundary_acc": round(eacc, 4),
            "mean_pred": round(statistics.fmean(ps), 3)}


def calibrate(ps, tv, grid=(0.30, 0.50, 0.60, 0.80)):
    """Sweep the two band thresholds on VALID only. Returns (t1, t2, best_acc)."""
    best, bt = -1, (0.40, 0.70)
    for t1 in np.arange(grid[0], grid[1], 0.01):
        for t2 in np.arange(grid[2], grid[3], 0.01):
            if t2 <= t1 + 0.05: continue
            acc = sum((0 if p < t1 else 1 if p < t2 else 2) == v for p, v in zip(ps, tv)) / len(ps)
            if acc > best: best, bt = acc, (round(float(t1), 2), round(float(t2), 2))
    return bt[0], bt[1], round(best, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out_distil")
    ap.add_argument("--eval", default=None, help="test | real | valid")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2) if dev == "cpu" else 8)

    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(a.out, exist_ok=True)

    if a.eval:  # evaluation-only path
        ck = torch.load(a.ckpt or f"{a.out}/best.pt", map_location=dev)
        m = Student(); m.load_state_dict(ck["model"]); m.to(dev)
        dl = DataLoader(PairDS(load(f"{a.data}/{a.eval}.jsonl"), tok), batch_size=64,
                        collate_fn=lambda b: collate(b, tok))
        ps, pv, ts, tv = predict(m, dl, dev)
        r = metrics(ps, pv, ts, tv, ck.get("thr", (0.40, 0.70, 0.50)))
        r["split"] = a.eval; r["n"] = len(ps); r["thr"] = ck.get("thr")
        print(json.dumps(r, indent=2)); return

    tr, va = load(f"{a.data}/train.jsonl"), load(f"{a.data}/valid.jsonl")
    dl_tr = DataLoader(PairDS(tr, tok), batch_size=BATCH, shuffle=True,
                       collate_fn=lambda b: collate(b, tok))
    dl_va = DataLoader(PairDS(va, tok), batch_size=64, collate_fn=lambda b: collate(b, tok))

    model = Student().to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"student params: {n/1e6:.1f}M  max_len={MAX_LEN}", flush=True)
    cw = torch.tensor([1.0, 0.97, 0.83], device=dev)  # from train verdict counts 355/366/426
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total = len(dl_tr) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    best, bad = float("inf"), 0
    for ep in range(1, EPOCHS + 1):
        model.train(); se, cnt = 0.0, 0
        for b in dl_tr:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=(dev == "cuda"), dtype=torch.float16):
                s, v = model(b["input_ids"], b["attention_mask"])
                loss = W_S * F.mse_loss(s, b["score"]) + W_V * F.cross_entropy(v, b["verdict"], weight=cw)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step(); opt.zero_grad()
            with torch.no_grad():
                se += F.mse_loss(s.float(), b["score"], reduction="sum").item(); cnt += len(s)
        ps, pv, ts, tv = predict(model, dl_va, dev)
        mae = statistics.fmean(abs(p - t) for p, t in zip(ps, ts))
        print(f"epoch {ep}: train MAE {se/cnt:.4f} | valid MAE {mae:.4f} (best {best:.4f})", flush=True)
        if mae < best - 1e-4:
            best, bad = mae, 0
            t1, t2, bacc = calibrate(ps, tv)
            torch.save({"model": model.state_dict(), "tok": MODEL, "arch": "distilbert",
                        "mae": mae, "thr": (t1, t2, 0.50), "band_acc_cal": bacc,
                        "max_len": MAX_LEN}, f"{a.out}/best.pt")
            print(f"  saved (thr={t1}/{t2}, cal band acc {bacc})", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE: print("early stop"); break
    print("best valid MAE:", round(best, 4))


def collate(b, tok):
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    L = max(len(x["input_ids"]) for x in b)
    return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (L - len(x["input_ids"])) for x in b]),
            "attention_mask": torch.tensor([x["attention_mask"] + [0] * (L - len(x["attention_mask"])) for x in b]),
            "score": torch.stack([x["score"] for x in b]),
            "verdict": torch.stack([x["verdict"] for x in b])}


if __name__ == "__main__":
    main()
