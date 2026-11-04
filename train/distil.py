"""DistilBERT student: fine-tune on rubric labels, tuned for CPU serving.

    python train/distil.py --data data --out out_distil                    # train
    python train/distil.py --data data --out out_distil --eval real        # eval one split
    python train/distil.py --data data --out out_distil --train-all        # final model, all rows

Measured on target box (8-core Kaby Lake, 4 threads): 35 ms/pair @ 72 tok, 270 MB RSS.
Real token lengths: p50=72 p99=118 max=137 (train), so max_len=192 covers 100%.

Anti-overfit defaults (1571 rows vs 66M params is the real constraint):
  * embeddings frozen           23.4M params — the most memorisation-prone block
  * bottom 2 of 6 layers frozen  ~14M params
  -> ~29M trainable instead of 66M

Compression fix: regression heads pull toward the mean on unseen candidates, which is why
threshold sweeps pin to the grid edge. We fit a linear recalibration (p' = a*p + b) on
VALID first, then sweep thresholds on the recalibrated scale.
"""
from __future__ import annotations
import argparse, json, os, random, statistics
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

MODEL = "distilbert-base-uncased"
HIDDEN = 768
MAX_LEN = 192
TEMPLATE = "Premise: {resume}\nHypothesis: this candidate matches the job: {job}\nAnswer: score"
BATCH, LR, WD, EPOCHS, PATIENCE, W_S, W_V, SEED = 32, 3e-5, 0.01, 8, 2, 0.7, 0.3, 4242
FREEZE_EMB, FREEZE_LAYERS = True, 2


def split_row(t: str):
    a, b = t.split("\n[SEP]\n", 1) if "\n[SEP]\n" in t else t.split("[SEP]", 1)
    a, b = a.strip(), b.strip()
    if a.startswith("resume:"): a = a[7:].strip()
    if b.startswith("job:"): b = b[3:].strip()
    return a, b


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; a, b = split_row(r["text"])
        enc = self.tok(TEMPLATE.format(resume=a, job=b), truncation=True, max_length=MAX_LEN)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "score": torch.tensor(r["score"], dtype=torch.float32),
                "verdict": torch.tensor(r["verdict"], dtype=torch.long)}


class Student(nn.Module):
    def __init__(self, hidden=HIDDEN, freeze_emb=FREEZE_EMB, freeze_layers=FREEZE_LAYERS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL)
        self.score = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.verdict = nn.Linear(hidden, 3)
        if freeze_emb:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False
        for i, layer in enumerate(self.encoder.transformer.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = h[:, 0]
        return torch.sigmoid(self.score(cls)).squeeze(-1), self.verdict(cls)


def load(f): return [json.loads(l) for l in open(f) if l.strip()]


def collate(b, tok):
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    L = max(len(x["input_ids"]) for x in b)
    return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (L - len(x["input_ids"])) for x in b]),
            "attention_mask": torch.tensor([x["attention_mask"] + [0] * (L - len(x["attention_mask"])) for x in b]),
            "score": torch.stack([x["score"] for x in b]),
            "verdict": torch.stack([x["verdict"] for x in b])}


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


def fit_recal(ps, ts, kind="isotonic"):
    """Map raw predictions to the label scale. Isotonic bends the curve, which is what
    fixes the model's refusal to predict extremes (pred_p10 0.19 vs true_p10 0.07)."""
    p, t = np.asarray(ps, dtype=float), np.asarray(ts, dtype=float)
    if kind == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, t)
            return ("iso", iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist())
        except ImportError:
            pass
    A = np.vstack([p, np.ones_like(p)]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return ("lin", float(a), float(b))


def apply_recal(ps, rec):
    if not rec:
        return list(ps)
    if rec[0] == "iso":
        return list(np.interp(ps, rec[1], rec[2]))
    _, a, b = rec
    return [min(1.0, max(0.0, a * p + b)) for p in ps]


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
    q = lambda xs, x: sorted(xs)[min(len(xs) - 1, int(x * len(xs)))]
    return {"mae": round(mae, 4), "band_acc": round(acc, 4), "apply_f1": round(f1, 4),
            "spearman": round(sp, 4),
            "pred_p10": round(q(ps, .1), 3), "pred_p50": round(q(ps, .5), 3), "pred_p90": round(q(ps, .9), 3),
            "true_p10": round(q(ts, .1), 3), "true_p50": round(q(ts, .5), 3), "true_p90": round(q(ts, .9), 3),
            "pred_sd": round(statistics.pstdev(ps), 3), "true_sd": round(statistics.pstdev(ts), 3)}


def calibrate(ps, tv):
    best, bt = -1, (0.40, 0.70)
    for t1 in np.arange(0.20, 0.55, 0.01):
        for t2 in np.arange(0.55, 0.90, 0.01):
            if t2 <= t1 + 0.05: continue
            acc = sum((0 if p < t1 else 1 if p < t2 else 2) == v for p, v in zip(ps, tv)) / len(ps)
            if acc > best: best, bt = acc, (round(float(t1), 2), round(float(t2), 2))
    return bt[0], bt[1], round(best, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out_distil")
    ap.add_argument("--eval", default=None); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--train-all", action="store_true", help="train on train+valid+test")
    ap.add_argument("--extra", default=None, help="comma-separated extra training jsonl (domain adaptation)")
    ap.add_argument("--calib", default=None, help="jsonl to fit recal+thresholds on (default: valid)")
    ap.add_argument("--no-freeze", action="store_true")
    ap.add_argument("--freeze-layers", type=int, default=FREEZE_LAYERS)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 4) if dev == "cpu" else 8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    os.makedirs(a.out, exist_ok=True)

    if a.eval:
        ck = torch.load(a.ckpt or f"{a.out}/best.pt", map_location=dev)
        m = Student(freeze_emb=False, freeze_layers=0)
        m.load_state_dict(ck["model"]); m.to(dev)
        dl = DataLoader(PairDS(load(f"{a.data}/{a.eval}.jsonl"), tok), batch_size=64,
                        collate_fn=lambda b: collate(b, tok))
        ps, pv, ts, tv = predict(m, dl, dev)
        if ck.get("recal"):
            ps = apply_recal(ps, ck["recal"])
        r = metrics(ps, pv, ts, tv, ck.get("thr", (0.40, 0.70, 0.50)))
        r.update(split=a.eval, n=len(ps), thr=ck.get("thr"), recal=ck.get("recal"),
                 ckpt_mae=round(ck.get("mae", -1), 4))
        print(json.dumps(r, indent=2)); return

    tr = load(f"{a.data}/train.jsonl"); va = load(f"{a.data}/valid.jsonl")
    if a.extra:
        for f in a.extra.split(","):
            ex = load(f.strip())
            print(f"  + {len(ex)} rows from {f.strip()}", flush=True)
            tr = tr + ex
    if a.calib:
        va = load(a.calib)
        print(f"calibrating on {a.calib} ({len(va)} rows)", flush=True)
    if a.train_all:
        tr = tr + va + load(f"{a.data}/test.jsonl")
        print(f"train-all: {len(tr)} rows (valid/test consumed — eval only on real.jsonl)", flush=True)
    dl_tr = DataLoader(PairDS(tr, tok), batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tok))
    dl_va = DataLoader(PairDS(va, tok), batch_size=64, collate_fn=lambda b: collate(b, tok))

    model = Student(freeze_emb=not a.no_freeze, freeze_layers=a.freeze_layers).to(dev)
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"student {tot/1e6:.1f}M total, {trn/1e6:.1f}M trainable (freeze_emb={not a.no_freeze}, freeze_layers={a.freeze_layers})", flush=True)

    cnt = Counter(r["verdict"] for r in tr)
    tot = sum(cnt.values())
    cw = torch.tensor([tot / max(1, cnt.get(i, 1)) for i in range(3)], dtype=torch.float32, device=dev)
    cw = cw.clamp(max=10.0)          # 16 v2 rows in real_train -> raw weight ~75 would explode
    cw = cw / cw.sum() * 3
    print(f"train verdicts {dict(sorted(cnt.items()))} -> class weights {[round(float(x),2) for x in cw]}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
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
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            with torch.no_grad():
                se += F.mse_loss(s.float(), b["score"], reduction="sum").item(); cnt += len(s)
        ps, pv, ts, tv = predict(model, dl_va, dev)
        ab = fit_recal(ps, ts)
        ps_r = apply_recal(ps, ab)
        mae = statistics.fmean(abs(p - t) for p, t in zip(ps_r, ts))
        print(f"epoch {ep}: train MAE {se/cnt:.4f} | valid MAE {mae:.4f} raw {statistics.fmean(abs(p-t) for p,t in zip(ps,ts)):.4f} (best {best:.4f})", flush=True)
        if mae < best - 1e-4:
            best, bad = mae, 0
            t1, t2, bacc = calibrate(ps_r, tv)
            torch.save({"model": model.state_dict(), "tok": MODEL, "arch": "distilbert", "mae": mae,
                        "thr": (t1, t2, 0.50), "recal": ab, "band_acc_cal": bacc,
                        "max_len": MAX_LEN, "train_all": a.train_all}, f"{a.out}/best.pt")
            tag = f"iso[{len(ab[1])}]" if ab[0] == "iso" else f"lin({ab[1]:.3f},{ab[2]:.3f})"
            print(f"  saved recal={tag} thr={t1}/{t2} band_acc={bacc}", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE: print("early stop"); break
    print("best valid MAE:", round(best, 4))


if __name__ == "__main__":
    main()
