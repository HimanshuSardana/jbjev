"""Split data/real.jsonl by JD into real_train / real_eval.

Grouped by job text, not by row: every JD in real_eval is a posting the model has
never seen, so the eval measures domain shift rather than memorisation of a JD.

    python tools/split_real.py --data data --eval-frac 0.2
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{a.data}/real.jsonl") if l.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[r["text"].split("[SEP]\njob: ")[-1]].append(r)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)

    n_eval = max(1, round(len(keys) * a.eval_frac))
    eval_jds, train_jds = set(keys[:n_eval]), set(keys[n_eval:])
    tr = [r for k in train_jds for r in groups[k]]
    ev = [r for k in eval_jds for r in groups[k]]
    random.Random(a.seed).shuffle(tr); random.Random(a.seed + 1).shuffle(ev)

    for name, rs in (("real_train", tr), ("real_eval", ev)):
        with open(f"{a.data}/{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        v = defaultdict(int)
        for r in rs:
            v[r["verdict"]] += 1
        jds = {r["text"].split("[SEP]\njob: ")[-1] for r in rs}
        print(f"{name}: {len(rs):5} rows | {len(jds):3} JDs | verdicts {dict(sorted(v.items()))}")
    print(f"eval JDs are disjoint from train JDs: {not (eval_jds & train_jds)}")


if __name__ == "__main__":
    main()
