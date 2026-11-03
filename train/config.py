# jev-like matcher on LFM3-1.7B-Base — training files
#
#   train/
#     config.py        single place for every knob (model id, LoRA, loss weights, paths)
#     data.py          JSONL -> premise/hypothesis cross-encoder inputs
#     model.py         frozen LFM3 + LoRA + score/verdict heads
#     train.py         LoRA fine-tune on Kaggle, early stop on valid MAE
#     evaluate.py      test + real eval: MAE, band acc, Spearman, calibration curve data
#     kaggle_jev.py    SINGLE FILE: paste into a Kaggle notebook cell and run. Same logic, no imports from this repo.
#
# Data contract (see data/README.md): score is source of truth.
#   score < 0.40 -> v0 | 0.40-0.70 -> v1 | >= 0.70 -> v2 | >= 0.50 -> apply 1

MODEL_ID = "LiquidAI/LFM3-1.7B-Base"   # Apache? No — lfm1.0. Read LICENSE before shipping.
FALLBACK_ID = "Qwen/Qwen3-1.7B-Base"   # one-line swap if LFM3 custom code fights you

# LFM3 facts from config.json (do not guess these):
#   hidden_size 2048, 24 layers (13 conv + 5 sliding + 6 full), vocab 128000,
#   tie_word_embeddings=true, bf16 weights, needs transformers>=5 + trust_remote_code=True
HIDDEN_SIZE = 2048
MAX_LEN = 512              # rows are 90-160 tok; 512 leaves headroom for long scraped JDs
TEMPLATE = "Premise: {resume}\nHypothesis: this candidate matches the job: {job}\nAnswer: score"

# LoRA — attention + short-conv linears only. MLP (w1/w2/w3) stays frozen.
# Full list available: q_proj k_proj v_proj o_proj gate_proj in_proj out_proj w1 w2 w3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]

LOSS_SCORE_W = 0.7
LOSS_VERDICT_W = 0.3

BATCH = 8                  # T4-16GB safe at seq 512 + grad-checkpointing + fp16
GRAD_ACCUM = 4             # effective batch 32
LR = 2e-4                  # LoRA LR (higher than full-finetune 3e-5)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EPOCHS = 5
PATIENCE = 2               # early stop on valid MAE

SEED = 4242
