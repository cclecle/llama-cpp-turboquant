#!/usr/bin/env python3
# rungsweep.py <port> <log> [frac] : run EVERY rung of the router's preset store once.
# Per rung: load, one chat request whose prompt fills `frac` (default 0.5) of the per-slot context, 256 greedy
# tokens, VRAM after generation (peak proxy, max over the cards), a gibberish check, unload.
# Appends one line per rung to <log>; rungs already in <log> are skipped, so a crashed sweep resumes.
import json, re, subprocess, sys, time, urllib.request

port, log = sys.argv[1], sys.argv[2]
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
base = "http://127.0.0.1:%s" % port
CHARS_PER_TOKEN = 3.2  # C++ source; the real prompt_n is printed

def req(p, b=None, t=600):
    d = json.dumps(b).encode() if b is not None else None
    r = urllib.request.Request(base + p, data=d, headers={"Content-Type": "application/json"} if d else {})
    with urllib.request.urlopen(r, timeout=t) as resp:
        return json.load(resp)

def vram():
    out = subprocess.run("rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used' | head -2 | grep -oE '[0-9]+$'",
                         shell=True, capture_output=True, text=True).stdout.split()
    return max(int(x) for x in out) / 1e9 if out else float("nan")

def gibberish(txt):
    words = txt.split()
    if len(words) < 20:
        return "SHORT(%d)" % len(words)
    wordy = sum(1 for w in words if re.search(r"[A-Za-z0-9一-鿿]", w)) / len(words)
    grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
    distinct = len(set(grams)) / max(1, len(grams))
    flags = []
    if wordy < 0.85: flags.append("nonword=%.2f" % (1 - wordy))
    if "�" in txt: flags.append("replchar")
    if distinct < 0.5: flags.append("repetitive=%.2f" % distinct)
    return "OK" if not flags else "CHECK:" + ",".join(flags)

def arg(a, k, dv=None):
    return a[a.index(k) + 1] if k in a else dv

done = set()
try:
    for l in open(log):
        if l.startswith("RUNG "):
            done.add(l.split()[1])
except FileNotFoundError:
    pass

corpus = open("/root/ppl.txt", errors="ignore").read()
models = req("/models")["data"]
out = open(log, "a")
for m in models:
    mid = m["id"]
    if mid in done:
        continue
    a = m["status"].get("args", [])
    per_slot = int(arg(a, "--ctx-size", 0)) // max(1, int(arg(a, "--parallel", 1)))
    t0 = time.time()
    line = ""
    try:
        for x in req("/models")["data"]:
            if x["status"]["value"] == "loaded":
                req("/models/unload", {"model": x["id"]}); time.sleep(3)
        req("/models/load", {"model": mid}, 60)
        st = "loading"
        while time.time() - t0 < 900:
            st = [x for x in req("/models")["data"] if x["id"] == mid][0]["status"]["value"]
            if st == "loaded" or (st == "unloaded" and time.time() - t0 > 45):
                break
            time.sleep(2)
        tl = time.time() - t0
        if st != "loaded":
            line = "RUNG %s FAIL load state=%s after %.0fs vram=%.2f" % (mid, st, tl, vram())
        else:
            low = mid.lower()
            if "embed" in low:
                j = req("/v1/embeddings", {"model": mid, "input": corpus[20000:22000]}, 900)
                emb = j["data"][0]["embedding"]
                ok = all(abs(x) < 1e30 for x in emb)
                line = "RUNG %s embed dim=%d load %.0fs vram %.2f %s" % (mid, len(emb), tl, vram(), "OK" if ok else "CHECK:nonfinite")
            elif "rerank" in low:
                docs = ["The capital of France is Paris.", "A recipe for chocolate cake.", "Quarterly sales figures for 2024."]
                j = req("/v1/rerank", {"model": mid, "query": "What is the capital of France?", "documents": docs, "top_n": 3}, 900)
                top = sorted(j["results"], key=lambda r: -r["relevance_score"])[0]["index"]
                line = "RUNG %s rerank top=%d load %.0fs vram %.2f %s" % (mid, top, tl, vram(), "OK" if top == 0 else "CHECK:rank")
            else:
                n_chars = int(per_slot * frac * CHARS_PER_TOKEN)
                prompt = corpus[10000:10000 + n_chars] + "\n\nSummarise the code above in five bullet points, then write a four-line poem about it."
                j = req("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": prompt}],
                                                 "max_tokens": 256, "temperature": 0}, 7200)
                tm = j.get("timings", {})
                msg = j["choices"][0]["message"]
                txt = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
                line = "RUNG %s ctx/slot=%d pp n=%d %.0f t/s tg n=%d %.1f t/s acc %s/%s load %.0fs vram %.2f %s | %s" % (
                    mid, per_slot, tm.get("prompt_n", 0), tm.get("prompt_per_second", 0), tm.get("predicted_n", 0),
                    tm.get("predicted_per_second", 0), tm.get("draft_n_accepted"), tm.get("draft_n"), tl, vram(),
                    gibberish(txt), re.sub(r"\s+", " ", txt)[:120])
        req("/models/unload", {"model": mid}); time.sleep(3)
    except Exception as e:
        line = "RUNG %s FAIL %s after %.0fs vram=%.2f" % (mid, str(e)[:200].replace("\n", " "), time.time() - t0, vram())
        try:
            req("/models/unload", {"model": mid})
        except Exception:
            pass
        time.sleep(5)
    out.write(line + "\n"); out.flush()
    print(line, flush=True)
out.write("SWEEP_DONE\n")
