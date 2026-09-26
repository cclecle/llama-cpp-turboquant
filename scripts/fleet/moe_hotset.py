#!/usr/bin/env python3
# moe_hotset.py : hot-expert placement file for the <device>_TIERED buffer type (--moe-hot-experts FILE).
#
#   moe_hotset.py --model M.gguf --counts SRC [--eval SRC] (--per-layer N | --match-uva N | --budget-gib G)
#                 [--policy uniform|greedy] -o hot.txt
#
# SRC is where the route counts per (layer, expert) come from:
#   - an imatrix GGUF written by llama-imatrix (tensors "blk.<l>.ffn_up_exps.weight.counts"),
#   - R9V's catalog.json ("ranking.training_counts"; with "heldout:" prefix, "ranking.heldout_counts").
# --eval SRC scores the placement on other counts (held-out) and prints the share of routes served from VRAM.
# Budgets: --per-layer N hot experts in every layer; --match-uva N the VRAM of today's rung with the experts of
# the first N layers in mapped host memory; --budget-gib G a total (all devices) of expert bytes in VRAM.
# --policy greedy spends the budget on the globally most-routed experts per byte instead of N per layer.
import argparse, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(here, '..', '..', 'gguf-py'),):
    if os.path.isdir(p):
        sys.path.insert(0, p)
from gguf import GGUFReader  # noqa: E402

KINDS = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'ffn_gate_up_exps')


def expert_bytes(model):
    """per layer: bytes of one expert across its gate/up/down tensors (all shards of a split GGUF)"""
    paths = [model]
    if '-00001-of-' in model:
        n = int(model.split('-of-')[1].split('.')[0])
        paths = [model.replace('-00001-of-', '-%05d-of-' % i) for i in range(1, n + 1)]
    per, n_exp = {}, None
    for p in paths:
        for t in GGUFReader(p).tensors:
            parts = t.name.split('.')
            if len(parts) >= 3 and parts[0] == 'blk' and parts[2] in KINDS:
                layer = int(parts[1])
                e = int(t.shape[-1]) if len(t.shape) == 3 else int(t.shape[2])
                n_exp = e
                per[layer] = per.get(layer, 0) + int(t.n_bytes) // e
    return per, n_exp


def load_counts(src):
    """{layer: [count per expert]}"""
    held = src.startswith('heldout:')
    if held:
        src = src[len('heldout:'):]
    if src.endswith('.json'):
        c = json.load(open(src))
        r = c.get('ranking', c)
        rows = r['heldout_counts' if held else 'training_counts']
        return {l: [float(x) for x in row] for l, row in enumerate(rows)}
    out = {}
    for t in GGUFReader(src).tensors:
        if t.name.endswith('.counts') and '_exps' in t.name:
            layer = int(t.name.split('.')[1])
            vals = [float(x) for x in t.data.reshape(-1)]
            # gate, up and down see the same routes; keep one of them
            out.setdefault(layer, vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--counts', required=True)
    ap.add_argument('--eval')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--per-layer', type=int)
    g.add_argument('--match-uva', type=int)
    g.add_argument('--budget-gib', type=float)
    ap.add_argument('--policy', choices=('uniform', 'greedy'), default='uniform')
    ap.add_argument('-o', '--out', required=True)
    a = ap.parse_args()

    eb, n_exp = expert_bytes(a.model)
    layers = sorted(eb)
    cnt = load_counts(a.counts)
    missing = [l for l in layers if l not in cnt]
    if missing:
        sys.exit('no route counts for layers %s' % missing)

    if a.per_layer is not None:
        budget = sum(eb[l] * a.per_layer for l in layers)
    elif a.match_uva is not None:
        budget = sum(eb[l] * n_exp for l in layers if l >= a.match_uva)
    else:
        budget = int(a.budget_gib * 2**30)

    order = {l: sorted(range(n_exp), key=lambda e: (-cnt[l][e], e)) for l in layers}
    hot = {l: [] for l in layers}
    if a.policy == 'uniform':
        n = a.per_layer if a.per_layer is not None else None
        if n is None:  # the most experts per layer that fit the budget
            n = min(n_exp, int(budget // sum(eb[l] for l in layers)))
        for l in layers:
            hot[l] = order[l][:n]
    else:
        cand = sorted(((cnt[l][e] / eb[l], l, e) for l in layers for e in range(n_exp)), reverse=True)
        used = 0
        for _, l, e in cand:
            if used + eb[l] > budget:
                continue
            hot[l].append(e)
            used += eb[l]

    used = sum(eb[l] * len(hot[l]) for l in layers)

    def served(counts):
        tot = sum(sum(counts[l]) for l in layers)
        return sum(sum(counts[l][e] for e in hot[l]) for l in layers) / tot if tot else float('nan')

    with open(a.out, 'w') as f:
        f.write('# moe_hotset.py %s\n' % ' '.join(sys.argv[1:]))
        f.write('# %d experts/layer, hot %d of %d (%.2f GiB of %.2f GiB expert bytes, all devices)\n' % (
            n_exp, sum(len(h) for h in hot.values()), n_exp * len(layers), used / 2**30,
            sum(eb[l] * n_exp for l in layers) / 2**30))
        for l in layers:
            f.write('blk.%d %s\n' % (l, ' '.join(map(str, sorted(hot[l])))))
    print('%s: hot %d/%d experts, %.2f GiB in VRAM (budget %.2f GiB), routes served from VRAM: %.1f%% (counts)' % (
        a.out, sum(len(h) for h in hot.values()), n_exp * len(layers), used / 2**30, budget / 2**30, 100 * served(cnt)), end='')
    if a.eval:
        print(', %.1f%% (%s)' % (100 * served(load_counts(a.eval)), a.eval))
    else:
        print()


if __name__ == '__main__':
    main()
