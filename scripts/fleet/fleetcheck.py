#!/usr/bin/env python3
# Fleet smoke check against one llama-server started with --models-preset <store>/main.ini --models-max 1.
# For every model id: load through a chat request (2k-token prompt, 512-token answer, greedy), print the
# timings, the answer head and a gibberish heuristic. Embedding / rerank models use their own endpoints.
#   fleetcheck.py <port> <model-id>...
# Reconstructed 2026-09-12 from the original's first half and its output format (the original was lost).
import json, os, sys, time, urllib.request, re, subprocess

port = sys.argv[1]; ids = sys.argv[2:]
base = "http://127.0.0.1:%s" % port

def req(path, body, timeout):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.load(resp)

t0 = time.time()
while time.time() - t0 < 300:
    try:
        urllib.request.urlopen(base + "/health", timeout=3); break
    except Exception:
        time.sleep(1)
else:
    print("SERVER NEVER READY"); sys.exit(1)

corpus = open("/root/ppl.txt", errors="ignore").read()
prompt_text = corpus[20000:20000 + 8000]   # ~2k tokens of C++ source

def vram():
    out = subprocess.run("rocm-smi --showmeminfo vram 2>/dev/null | grep -i Used | head -2 | grep -oE '[0-9]+$'", shell=True, capture_output=True, text=True).stdout.split()
    return " ".join("%.2f" % (int(v)/1e9) for v in out)

def gibberish(txt):
    words = txt.split()
    if len(words) < 20:
        return "SHORT(%d words)" % len(words)
    wordy = sum(1 for w in words if re.search(r"[A-Za-z0-9一-鿿]", w)) / len(words)
    bad = txt.count("�")
    grams = [tuple(words[i:i+5]) for i in range(len(words) - 4)]
    distinct = len(set(grams)) / max(1, len(grams))
    flags = []
    if wordy < 0.85: flags.append("nonword=%.2f" % (1 - wordy))
    if bad: flags.append("replacement_chars=%d" % bad)
    if distinct < 0.5: flags.append("repetitive(distinct5gram=%.2f)" % distinct)
    return "OK" if not flags else "CHECK " + " ".join(flags)

for mid in ids:
    t = time.time()
    try:
        low = mid.lower()
        if "embedding" in low or "embed" in low:
            j = req("/v1/embeddings", {"model": mid, "input": prompt_text[:2000]}, 600)
            emb = j["data"][0]["embedding"]
            finite = all(abs(x) < 1e30 for x in emb)
            print("%-42s embeddings dim=%d finite=%s load+run %ds vram %s -> %s" % (mid, len(emb), finite, time.time() - t, vram(), "OK" if finite else "FAIL"))
        elif "rerank" in low:
            docs = ["The capital of France is Paris.", "A recipe for chocolate cake.", "Quarterly sales figures for 2024."]
            j = req("/v1/rerank", {"model": mid, "query": "What is the capital of France?", "documents": docs, "top_n": 3}, 600)
            res = sorted(j["results"], key=lambda r: -r["relevance_score"])
            scores = ["%.2f" % r["relevance_score"] for r in res]
            print("%-42s rerank top=%d scores=%s load+run %ds vram %s -> %s" % (mid, res[0]["index"], scores, time.time() - t, vram(), "OK" if res[0]["index"] == 0 else "CHECK"))
        else:
            j = req("/v1/chat/completions", {"model": mid, "max_tokens": 512, "temperature": 0, "messages": [
                {"role": "user", "content": prompt_text + "\n\nSummarise the code above in five bullet points, then write a four-line poem about it."}]}, 900)
            tm = j.get("timings", {}); txt = (j["choices"][0]["message"].get("content") or "").strip()
            print("%-42s pp n=%d %.0f t/s | tg n=%d %.1f t/s | load+run %ds vram %s -> %s" % (
                mid, tm.get("prompt_n", 0), tm.get("prompt_per_second", 0), tm.get("predicted_n", 0), tm.get("predicted_per_second", 0), time.time() - t, vram(), gibberish(txt)))
            print("    text: " + re.sub(r"\s+", " ", txt)[:300])
    except Exception as e:
        print("%-42s FAIL %s load+run %ds vram %s" % (mid, str(e)[:160], time.time() - t, vram()))
    sys.stdout.flush()
print("FLEET_DONE")
