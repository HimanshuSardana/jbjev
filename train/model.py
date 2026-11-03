"""Frozen LFM3 + LoRA + two heads. Cross-encoder: one sequence, read last-token hidden."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .config import HIDDEN_SIZE


class JevHead(nn.Module):
    def __init__(self, hidden: int = HIDDEN_SIZE, drop: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 1),
        )
        self.verdict = nn.Linear(hidden, 3)

    def forward(self, h: torch.Tensor):
        return torch.sigmoid(self.score(h)).squeeze(-1), self.verdict(h)


class JevModel(nn.Module):
    """Backbone returns hidden states; we pool the last non-pad token (causal LM)."""

    def __init__(self, backbone, head: JevHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        h = out.hidden_states[-1]                       # (B, T, H)
        idx = attention_mask.sum(1).clamp_min(1).sub(1).long()  # last real token
        last = h[torch.arange(h.size(0), device=h.device), idx].float()  # heads are fp32
        return self.head(last)


def build_backbone(model_id: str, *, gradient_checkpointing: bool = True):
    """Load LFM3 (custom code) in fp16 for T4. Freeze base; embeddings stay frozen (tied)."""
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float16)
    except TypeError:  # transformers < 4.56
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
    if gradient_checkpointing and hasattr(m, "gradient_checkpointing_enable"):
        m.gradient_checkpointing_enable()
    for p in m.parameters():
        p.requires_grad = False
    # keep LM head frozen too — we never generate
    return m


def apply_lora(backbone, r: int, alpha: int, dropout: float, targets: list[str]):
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha,
        lora_dropout=dropout, target_modules=targets, bias="none",
    )
    return get_peft_model(backbone, cfg)


@torch.no_grad()
def predict(texts, tokenizer, model, *, max_len: int = 512, batch: int = 8,
            device: str = "cuda") -> tuple[list[float], list[int], list[float]]:
    """score, verdict, confidence. confidence = 2*|score-0.5|, the jev-style typed answer."""
    import torch.nn.functional as F
    model.eval()
    scores, verdicts = [], []
    amp = torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.float16)
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i:i + batch], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        with amp:
            s, v = model(enc["input_ids"], enc["attention_mask"])
        scores += s.float().cpu().tolist()
        verdicts += v.argmax(-1).cpu().tolist()
    conf = [round(2 * abs(s - 0.5), 3) for s in scores]
    return scores, verdicts, conf
