"""Fast CPU inference: int8 dynamic quant + warmup + batched predict.

    from serve.infer import Matcher
    m = Matcher("out/best.pt")
    m.score([("resume text", "job text"), ...])   -> [{"score":..,"verdict":..,"apply":..,"confidence":..}]

Design notes:
  * int8 dynamic quant on nn.Linear — 3-4x faster matmul on CPU, ~2.5x smaller RSS.
    Accuracy cost on this head is small; verify with train/bench.py --int8 before shipping.
  * dynamic padding: we pad to the longest row in the batch, never to max_len.
  * torch.set_num_threads() pinned to physical cores; oversubscribing hurts.
  * warmup matters: first call pays lazy-init and quant kernel selection.
"""
from __future__ import annotations
import os
import torch

from train import config as C
from train.data import split_row, to_prompt
from train.model import JevHead, JevModel, build_backbone, apply_lora


class Matcher:
    """Loads either checkpoint type: LFM3 cross-encoder or the DistilBERT student."""

    def __init__(self, ckpt: str, model_id: str = C.MODEL_ID, int8: bool = True,
                 threads: int | None = None, max_len: int | None = None):
        # measured on 8-core Kaby Lake: 4 threads beats 8 (hyperthread contention)
        threads = threads or min(4, max(1, (os.cpu_count() or 4)))
        torch.set_num_threads(threads)
        torch.set_grad_enabled(False)

        from transformers import AutoTokenizer
        ck = torch.load(ckpt, map_location="cpu")
        mid = ck.get("tok", model_id)
        self.arch = ck.get("arch", "lfm3")
        self.thr = ck.get("thr", (0.40, 0.70, 0.50))   # calibrated on valid
        self.recal = ck.get("recal")                     # None | ("lin",a,b) | ("iso",xs,ys)
        self.max_len = max_len or ck.get("max_len", 192 if self.arch == "distilbert" else 256)

        self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=(self.arch == "lfm3"))
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        if self.arch == "distilbert":
            from train.distil import Student
            model = Student()
            model.load_state_dict(ck["model"])
        else:
            base = build_backbone(mid, gradient_checkpointing=False)
            base = apply_lora(base, C.LORA_R, C.LORA_ALPHA, C.LORA_DROPOUT, C.LORA_TARGETS)
            base.load_state_dict(ck["lora"], strict=False)
            head = JevHead(); head.load_state_dict(ck["head"])
            model = JevModel(base, head)
        model = model.eval()
        if int8:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.int8 = int8
        self._warm()

    def _warm(self):
        dummy = to_prompt("Python, SQL", "Backend Engineer. 2+ yrs.", C.TEMPLATE)
        for _ in range(2):
            self._forward([dummy])

    def _forward(self, prompts):
        enc = self.tok(prompts, truncation=True, max_length=self.max_len,
                       padding=True, return_tensors="pt")
        with torch.inference_mode():
            s, v = self.model(enc["input_ids"], enc["attention_mask"])
        return s.float().tolist(), v.argmax(-1).tolist()

    def score(self, pairs, batch: int = 16):
        """pairs: list of (resume, job) or raw 'resume: ...\\n[SEP]\\njob: ...' strings."""
        prompts = []
        for p in pairs:
            if isinstance(p, str):
                prompts.append(to_prompt(*split_row(p), C.TEMPLATE))
            else:
                prompts.append(to_prompt(p[0], p[1], C.TEMPLATE))
        out = []
        for i in range(0, len(prompts), batch):
            ss, vv = self._forward(prompts[i:i + batch])
            t1, t2, ta = self.thr
            import numpy as _np
            rec = self.recal
            for s, v in zip(ss, vv):
                if rec and rec[0] == "iso":
                    xs, ys = _np.asarray(rec[1]), _np.asarray(rec[2])
                    if s < xs[0]:
                        k = min(3, len(xs) - 1)
                        s = ys[0] + (ys[k] - ys[0]) / max(1e-6, xs[k] - xs[0]) * (s - xs[0])
                    elif s > xs[-1]:
                        k = min(3, len(xs) - 1)
                        s = ys[-1] + (ys[-1] - ys[-1 - k]) / max(1e-6, xs[-1] - xs[-1 - k]) * (s - xs[-1])
                    else:
                        s = float(_np.interp(s, xs, ys))
                    s = float(min(1.0, max(0.0, s)))
                elif rec:
                    s = min(1.0, max(0.0, rec[1] * s + rec[2]))
                out.append({
                    "score": round(s, 3),
                    "verdict": 0 if s < t1 else 1 if s < t2 else 2,
                    "apply": int(s >= ta),
                    "confidence": round(2 * abs(s - 0.50), 3),
                })
        return out


if __name__ == "__main__":
    import sys, time, json
    m = Matcher(sys.argv[1] if len(sys.argv) > 1 else "out/best.pt")
    demo = [
        ("Python, Django, Postgres, 3 yrs backend", "Backend Engineer. Python, Django, Postgres. 2+ yrs."),
        ("Photoshop, Illustrator, print design", "Backend Engineer. Python, Django, Postgres. 2+ yrs."),
    ]
    t0 = time.perf_counter()
    r = m.score(demo)
    dt = (time.perf_counter() - t0) * 1000
    print(json.dumps(r, indent=2))
    print(f"2 pairs in {dt:.0f} ms (int8={m.int8}, threads={torch.get_num_threads()})")
