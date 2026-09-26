#!/usr/bin/env python3
# step_ms.py <label prefix> [results.jsonl] : per-run decode cost in ms per speculative step, the stable metric of a
# decode A/B (tokens/s moves with the acceptance, which changes with the generated text from one load to the next).
# A step is one target verify pass plus its drafts; steps = generated tokens - accepted draft tokens.
import json, sys

prefix = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else 'results.jsonl'
arms = {}
for line in open(path):
    d = json.loads(line)
    if not d.get('label', '').startswith(prefix):
        continue
    s = d['server']
    steps = s['predicted_n'] - s['draft_n_accepted']
    ms = s['predicted_ms']/steps
    print('%-28s decode %6.2f t/s  %4d steps  %5.1f ms/step  %.2f tok/step  prefill %5.0f t/s' % (
        d['label'], d['decode_tps'], steps, ms, s['predicted_n']/steps, d['prefill_tps']))
    # arms are named <prefix><arm letter><run number>
    arm = d['label'][len(prefix):len(prefix) + 1]
    arms.setdefault(arm, []).append(ms)
for arm, v in sorted(arms.items()):
    print('arm %s: mean %.1f ms/step over %d runs' % (arm, sum(v)/len(v), len(v)))
