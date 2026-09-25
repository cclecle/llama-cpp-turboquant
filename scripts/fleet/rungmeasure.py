#!/usr/bin/env python3
# rungmeasure.py <port> <model-id>...  : load each rung through the router, 10k-token prompt + 256 gen,
# VRAM read AFTER generation (peak proxy), text sanity, then unload. Cap policy: 33.70-33.90 GB used.
import json, sys, time, urllib.request, urllib.error, subprocess, re
port, ids = sys.argv[1], sys.argv[2:]
base = "http://127.0.0.1:%s" % port
def req(p, b=None, t=600):
    d = json.dumps(b).encode() if b is not None else None
    r = urllib.request.Request(base + p, data=d, headers={"Content-Type": "application/json"} if d else {})
    with urllib.request.urlopen(r, timeout=t) as resp: return json.load(resp)
def status(mid):
    for m in req("/models")["data"]:
        if m["id"] == mid: return m["status"]["value"]
    return "missing"
def vram():
    # max over the visible cards (under tensor split both matter; the cap is per card)
    out = subprocess.run("rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used' | head -2 | grep -oE '[0-9]+$'", shell=True, capture_output=True, text=True).stdout.split()
    return max(int(x) for x in out) / 1e9 if out else float("nan")
corpus = open("/root/ppl.txt", errors="ignore").read()
prompt = corpus[30000:70000] + "\n\nSummarise the code above in five bullet points, then write a four-line poem about it."
for mid in ids:
    t0 = time.time()
    try:
        for m in req("/models")["data"]:
            if m["status"]["value"] == "loaded":
                req("/models/unload", {"model": m["id"]}); time.sleep(3)
        req("/models/load", {"model": mid}, 60)
        st = "loading"
        while time.time() - t0 < 600:
            st = status(mid)
            if st == "loaded": break
            if st == "unloaded" and time.time() - t0 > 45: break
            time.sleep(2)
        if st != "loaded":
            print("FAIL %-36s state=%s after %.0fs vram=%.2f" % (mid, st, time.time() - t0, vram())); continue
        v_load = vram(); tl = time.time() - t0
        j = req("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0})
        v_peak = vram(); tm = j.get("timings", {}); txt = j["choices"][0]["message"]["content"]
        words = txt.split(); bad = "SHORT" if len(words) < 12 else ("REPL" if "\ufffd" in txt else "OK")
        print("%-36s load %3.0fs  pp n=%5d %6.0f t/s  tg n=%3d %5.1f t/s  acc %s/%s  VRAM load %.2f  after-gen %.2f  free %.2f GB  %s" % (
            mid, tl, tm.get("prompt_n", 0), tm.get("prompt_per_second", 0), tm.get("predicted_n", 0), tm.get("predicted_per_second", 0),
            tm.get("draft_n_accepted"), tm.get("draft_n"), v_load, v_peak, 34.208743424 - v_peak, bad))
        print("    text: " + re.sub(r"\s+", " ", txt)[:160])
        req("/models/unload", {"model": mid}); time.sleep(3)
    except Exception as e:
        print("FAIL %-36s %s vram=%.2f" % (mid, str(e)[:160], vram()))
        try: req("/models/unload", {"model": mid})
        except Exception: pass
        time.sleep(3)
