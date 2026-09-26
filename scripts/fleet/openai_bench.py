#!/usr/bin/env python3
# openai_bench.py <base url> <model> <prompt file> <label> <results.jsonl> [max_tokens 2048] [warmup 1]
# One streaming chat completion against any OpenAI-compatible server (llama-server, vLLM), identical request for
# every backend: temperature 0, no ignore_eos, max_tokens as given. Records the time to first token (prefill),
# decode t/s over the generated tokens (usage counts, first token to last), the server's own timings when it
# sends them (llama-server "timings", incl. draft acceptance), and vLLM's spec-decode counters from /metrics.
# A short warmup request runs first so JIT/graph compilation is not billed to the measured prefill.
import json, sys, time, urllib.request

base, model, prompt_file, label, out = sys.argv[1:6]
max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 2048
warmup = int(sys.argv[7]) if len(sys.argv) > 7 else 1
prompt = open(prompt_file, encoding='utf-8', errors='ignore').read()


def metrics():
    try:
        txt = urllib.request.urlopen(base + '/metrics', timeout=10).read().decode()
    except Exception:
        return {}
    m = {}
    for line in txt.splitlines():
        if line.startswith('vllm:spec_decode_') and not line.startswith('#'):
            k, v = line.rsplit(' ', 1)
            k = k.split('{')[0]
            m[k] = m.get(k, 0.0) + float(v)
    return m


def run(text, n):
    body = {'model': model, 'messages': [{'role': 'user', 'content': text}], 'max_tokens': n, 'temperature': 0,
            'stream': True, 'stream_options': {'include_usage': True}}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time()
    t_first = t_last = None
    usage, timings, finish, n_chunks, chars = {}, {}, None, 0, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            d = json.loads(line[5:])
            usage = d.get('usage') or usage
            timings = d.get('timings') or timings
            for c in d.get('choices', []):
                delta = c.get('delta', {})
                piece = (delta.get('content') or '') + (delta.get('reasoning_content') or '') + (delta.get('reasoning') or '')
                if piece:
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    n_chunks += 1
                    chars += len(piece)
                finish = c.get('finish_reason') or finish
    return t0, t_first, t_last, usage, timings, finish, n_chunks, chars


if warmup:
    run('Say hello in five words.', 16)
m0 = metrics()
t0, t_first, t_last, usage, timings, finish, n_chunks, chars = run(prompt, max_tokens)
m1 = metrics()
pt, ct = usage.get('prompt_tokens'), usage.get('completion_tokens')
ttft = t_first - t0
dec = (ct - 1) / (t_last - t_first) if ct and ct > 1 and t_last > t_first else None
res = {'label': label, 'prompt_tokens': pt, 'completion_tokens': ct, 'finish': finish, 'ttft_s': round(ttft, 3),
       'prefill_tps': round(pt / ttft, 1) if pt else None, 'decode_tps': round(dec, 2) if dec else None,
       'total_s': round(t_last - t0, 2), 'chunks': n_chunks, 'chars': chars}
if timings:
    res['server'] = {k: timings.get(k) for k in ('prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n',
                                                 'predicted_ms', 'predicted_per_second', 'draft_n', 'draft_n_accepted')}
    if timings.get('draft_n'):
        res['accept'] = round(timings['draft_n_accepted'] / timings['draft_n'], 3)
spec = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
if spec:
    res['vllm_spec'] = spec
    drafts = spec.get('vllm:spec_decode_num_draft_tokens_total')
    if drafts:
        res['accept'] = round(spec.get('vllm:spec_decode_num_accepted_tokens_total', 0) / drafts, 3)
print(json.dumps(res))
open(out, 'a').write(json.dumps(res) + '\n')
