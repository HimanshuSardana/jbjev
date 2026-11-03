# Kaggle — jev-like matcher on LFM3-1.7B-Base
# Free tier: 30 GPU-hrs/week, T4-16GB, 12h sessions. This run: ~15 min/epoch, 3-5 epochs.
#
# Setup:
#   1. Kaggle notebook, GPU T4 x2 ON, internet ON
#   2. Add this repo's data/ as a Kaggle dataset (or `!git clone <repo>`), so data/*.jsonl exist
#   3. Cell 1:  !pip uninstall -y torchao && !pip install -q "transformers>=5" peft accelerate scikit-learn
#      (Kaggle's torchao 0.10 is below peft's required 0.16 and makes peft raise instead of
#       falling back to plain LoRA. We never use torchao — remove it.)
#   4. Cell 2:  paste THIS FILE (or !python train/kaggle_jev.py --data data --out /kaggle/working/out)
#
# What it does: frozen LFM3-Base + LoRA (attn + conv linears) + score/verdict heads,
# cross-encoder over "Premise/Hypothesis/Answer: score", early stop on valid MAE.
# Outputs /kaggle/working/out/best.pt — download it, thresholds ship beside it.
"""Single-file copy of train/ for Kaggle. Usage: python kaggle_jev.py --data data --out out [--model LiquidAI/LFM3-1.7B-Base]"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

MODEL_ID = "LiquidAI/LFM3-1.7B-Base"
HIDDEN = 2048  # from LFM3 config.json — do not change
MAX_LEN = 512
TEMPLATE = "Premise: {resume}\nHypothesis: this candidate matches the job: {job}\nAnswer: score"
R, ALPHA, DROP = 16, 32, 0.05
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]
W_S, W_V, LR, WD, BATCH, ACCUM, EPOCHS, PATIENCE, SEED = 0.7, 0.3, 5e-4, 0.01, 8, 4, 12, 3, 4242


def split_row(t: str):
    r, j = t.split("\n[SEP]\n", 1) if "\n[SEP]\n" in t else t.split("[SEP]", 1)
    return (r[len("resume:"):].strip() if r.strip().startswith("resume:") else r.strip()), j.strip()


class PairDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        resume, job = split_row(r["text"])
        enc = self.tok(TEMPLATE.format(resume=resume, job=job), truncation=True, max_length=MAX_LEN)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "score": torch.tensor(r["score"], dtype=torch.float32),
                "verdict": torch.tensor(r["verdict"], dtype=torch.long)}


class JevHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(HIDDEN, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
        self.verdict = nn.Linear(HIDDEN, 3)
    def forward(self, h):
        return torch.sigmoid(self.score(h)).squeeze(-1), self.verdict(h)


class JevModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, use_cache=False).hidden_states[-1]
        idx = attention_mask.sum(1).clamp_min(1).sub(1).long()
        pooled = h[torch.arange(h.size(0), device=h.device), idx].float()  # heads are fp32
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
    ap.add_argument("--model", default=MODEL_ID); ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| gpu:", torch.cuda.get_device_name(0) if dev == "cuda" else "none", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    load = lambda f: [json.loads(l) for l in open(f"{a.data}/{f}.jsonl") if l.strip()]
    tr, va = load("train"), load("valid")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def coll(b):
        L = max(len(x["input_ids"]) for x in b)
        return {"input_ids": torch.tensor([x["input_ids"] + [pad] * (L - len(x["input_ids"])) for x in b]),
                "attention_mask": torch.tensor([x["attention_mask"] + [0] * (L - len(x["attention_mask"])) for x in b]),
                "score": torch.stack([x["score"] for x in b]), "verdict": torch.stack([x["verdict"] for x in b])}
    dl_tr = DataLoader(PairDS(tr, tok), batch_size=BATCH, shuffle=True, collate_fn=coll)
    dl_va = DataLoader(PairDS(va, tok), batch_size=BATCH * 2, collate_fn=coll)

    print("loading backbone (fp16, frozen)...", flush=True)
    try:
        base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float16)
    except TypeError:  # transformers < 4.56
        base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.float16)
    if hasattr(base, "gradient_checkpointing_enable"): base.gradient_checkpointing_enable()
    for p in base.parameters(): p.requires_grad = False
    from peft import LoraConfig, get_peft_model, TaskType
    try:  # torchao below peft's floor makes the LoRA dispatcher raise instead of falling back
        import torchao  # noqa: F401
        from peft.import_utils import is_torchao_available
        if not is_torchao_available():
            raise ImportError("torchao incompatible")
    except ImportError:
        pass  # fine: plain LoRA path
    except Exception:
        raise SystemExit("torchao present but incompatible with peft. Run: pip uninstall -y torchao")
    base = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=R, lora_alpha=ALPHA,
                                           lora_dropout=DROP, target_modules=TARGETS, bias="none"))
    base.print_trainable_parameters()
    parents = {n.rsplit(".lora_A", 1)[0].rsplit(".", 1)[-1]
               for n, _ in base.named_modules() if ".lora_A" in n}
    print("LoRA parent modules:", sorted(parents), flush=True)
    n_conv = sum(1 for n, _ in base.named_modules()
                 if ".lora_A" in n and ("in_proj" in n or "out_proj" in n))
    print(f"LoRA on conv linears: {n_conv} (13 in_proj + 13 out_proj = 26)", flush=True)
    model = JevModel(base, JevHead()).to(dev)
    cw = torch.tensor([1147 / 355, 1147 / 366, 1147 / 426], dtype=torch.float32)
    cw = (cw / cw.sum() * 3).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
    total = len(dl_tr) * a.epochs // ACCUM
    sched = get_linear_schedule_with_warmup(opt, int(total * 0.1), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    os.makedirs(a.out, exist_ok=True)
    best, bad = float("inf"), 0
    for ep in range(1, a.epochs + 1):
        model.train(); opt.zero_grad()
        tr_se, tr_n = 0.0, 0
        for i, b in enumerate(dl_tr):
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=(dev == "cuda"), dtype=torch.float16):
                s, v = model(b["input_ids"], b["attention_mask"])
                loss = (W_S * F.mse_loss(s, b["score"]) + W_V * F.cross_entropy(v, b["verdict"], weight=cw)) / ACCUM
            with torch.no_grad():
                tr_se += F.mse_loss(s.float(), b["score"], reduction="sum").item(); tr_n += len(s)
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM == 0:
                prev = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= prev:   # skip the step if grads overflowed
                    sched.step()
                opt.zero_grad()
        model.eval(); se, n = 0.0, 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(dev == "cuda"), dtype=torch.float16):
            for b in dl_va:
                b = {k: v.to(dev) for k, v in b.items()}
                s, _ = model(b["input_ids"], b["attention_mask"])
                se += F.mse_loss(s.float(), b["score"], reduction="sum").item(); n += len(s)
        mae = (se / n) ** 0.5
        print(f"epoch {ep}: train MAE {tr_se / max(1, tr_n):.4f} | valid MAE {mae:.4f} (best {best:.4f})", flush=True)
        if mae < best - 1e-4:
            best, bad = mae, 0
            torch.save({"lora": base.state_dict(), "head": model.head.state_dict(), "tok": a.model, "mae": mae},
                       f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= PATIENCE: print("early stop"); break
    print("best valid MAE:", best)


if __name__ == "__main__":
    main()
