#!/usr/bin/env python3
# visiontest.py <port> <model-id> <image-file> [max_tokens] : load the rung, ask about the image, show reasoning vs answer, unload
import json, sys, time, base64, urllib.request, subprocess
port, mid, img = sys.argv[1:4]; maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 120
base = "http://127.0.0.1:%s" % port
def req(p, b=None, t=600):
    d = json.dumps(b).encode() if b is not None else None
    r = urllib.request.Request(base + p, data=d, headers={"Content-Type": "application/json"} if d else {})
    with urllib.request.urlopen(r, timeout=t) as resp: return json.load(resp)
for m in req("/models")["data"]:
    if m["status"]["value"] == "loaded": req("/models/unload", {"model": m["id"]}); time.sleep(3)
req("/models/load", {"model": mid}); t = time.time()
while time.time() - t < 300:
    st = [m for m in req("/models")["data"] if m["id"] == mid][0]["status"]["value"]
    if st == "loaded": break
    time.sleep(2)
b64 = base64.b64encode(open(img, "rb").read()).decode()
j = req("/v1/chat/completions", {"model": mid, "max_tokens": maxtok, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    {"type": "text", "text": "Describe this image in two sentences. What text, if any, does it contain?"}]}]})
vr = subprocess.run("rocm-smi --showmeminfo vram 2>/dev/null | grep -i Used | head -2 | grep -oE '[0-9]+$'", shell=True, capture_output=True, text=True).stdout.split()
msg = j["choices"][0]["message"]; fin = j["choices"][0].get("finish_reason"); tm = j.get("timings", {})
print("%s  image tokens pp n=%s  gen n=%s finish=%s  vram after %.2f GB" % (mid, tm.get("prompt_n"), tm.get("predicted_n"), fin, max(int(x) for x in vr) / 1e9))
print("    reasoning: %s" % str(msg.get("reasoning_content") or "")[:220].replace("\n", " "))
print("    answer:    %s" % (msg.get("content") or "").strip().replace("\n", " ")[:300])
req("/models/unload", {"model": mid})
