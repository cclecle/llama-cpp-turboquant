#!/usr/bin/env python3
"""
Performance and end-to-end validation for this fork.

Everything is measured through llama-server in router mode, against the same .ini production
reads. llama-bench was removed: it cannot read a preset, so its flags had to be hand-derived,
and a silently wrong derivation (f16 KV where the preset says q8_0) inverted a decode result and
nearly buried a real regression. It also cannot do speculative decoding, which every model in
this fleet uses. Faithfulness beats convenience.

Modes:
  smoke   load one preset section faithfully at production context, generate, record load + VRAM
  gate    one depth (16k), full basket, real preset - the fast pass over the risky models
  deep    two-turn conversation per model at ~50% then ~90% of its CONFIGURED context, through
          the server with speculation live - the depth the preset actually runs at

Reference handling:
  --baseline  writes results into reference.json (run this against the PRODUCTION build first)
  otherwise   compares against reference.json and exits non-zero on regression beyond tolerance
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Regression tolerances. Negative = "may not drop by more than". Deliberately conservative to
# start; tighten once run-to-run variance on this box is known.
TOLERANCE = {
    "prefill_tps":  -0.05,   # -5%
    "decode_tps":   -0.03,   # -3%
    "load_seconds": +0.10,   # +10% (regression is slower, so positive)
    "peak_vram_gb": +0.05,   # +5%
    "acceptance":   -0.02,   # absolute, not relative
}
HIGHER_IS_BETTER = {"prefill_tps", "decode_tps", "acceptance"}

# Fast gate subset: one depth, full basket, real preset. Chosen so the three questions most
# likely to fail get answered in minutes rather than hours:
GATE_MODELS = [
    "Qwen3.6-27B:HQ",                  # head_dim 256 - the rocWMMA removal verdict
    "Devstral-Small-2-24B-Insctruct",  # q6_K MMQ prefill + K-quant float matvec decode
    "GLM-4.7-Flash-30B-A3B",           # both RDNA4 MLA kernels (D=576)
]

# Everything the fleet actually cares about is measured through the server instead, with the real
# preset and speculation live.
DEEP_MODELS = [
    "Mistral-Medium-3.5-128B",   # pinned host KV, EAGLE, tensor split, tight VRAM fit
    "Qwen3.5-122B-A10B",         # CPU-resident experts under tensor split, MTP
    "Mistral-Small-4-119B-A6B",  # ONLY fleet model on the deepseek2 graph (double-prune merge)
    "Devstral-Small-2-24B-Insctruct",
    "Qwen3.6-27B:HQ",
    "GLM-4.7-Flash-30B-A3B",
    "Qwen3.6-35B-A3B:HQ",        # CPU-MoE + np=2
    "gemma-4-31B",               # SWA + Q4_K
]

DEPTHS = [8192, 16384, 32768]

# Minimum generation length for ANY measured run, smoke included.
#
# Short generations measure clock ramp, not steady-state decode: a 32-token sample on
# Mistral-Medium read 8.6 t/s against a production figure of ~15 mean / 26 peak. On this box the
# extra tokens cost seconds - less than the model load they follow - so there is no reason to
# measure anything shorter. It also matters for speculation: acceptance over 32 tokens is noise,
# and draft_n / draft_n_accepted only become meaningful over a few hundred.
N_PREDICT = 2048

# 3 code + 3 general. A single prompt is untrustworthy: a short Python prompt once showed a bogus
# +75 t/s spike that vanished on a 7k C++ prompt and on the basket average.
BASKET = [
    ("code", "Write a Python function that merges two sorted linked lists in place. Explain the invariant."),
    ("code", "Given a C++ struct with a raw pointer member, write a correct move constructor and assignment operator."),
    ("code", "Refactor this into a state machine: a parser that reads key=value pairs separated by semicolons."),
    ("gen",  "Explain why PCIe lane allocation matters for multi-GPU inference on consumer platforms."),
    ("gen",  "Summarise the trade-offs between quantising the KV cache and quantising model weights."),
    ("gen",  "Describe how speculative decoding stays lossless at temperature 0."),
]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- server control

class Server:
    """Starts llama-server in router mode against a preset file, on a private port.

    Router mode is used deliberately: it is the only way to load a model exactly as production
    does, including every key in the .ini section. Loading by CLI flags would test a
    configuration nobody runs.
    """

    def __init__(self, build, preset_ini, port, logpath, extra_env=None):
        self.build = build
        self.preset_ini = preset_ini
        self.port = port
        self.logpath = logpath
        self.proc = None
        self.extra_env = extra_env or {}

    def start(self, timeout=900):
        env = dict(os.environ)
        env.setdefault("HIP_VISIBLE_DEVICES", "0,1")   # hide the iGPU
        env.update(self.extra_env)
        cmd = [
            os.path.join(self.build, "bin", "llama-server"),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--models-preset", self.preset_ini,
            "--models-max", "1",
            "--no-models-autoload",
            "--slots",
        ]
        self.fh = open(self.logpath, "w")
        # own process group so we can kill only our tree, never production
        self.proc = subprocess.Popen(cmd, stdout=self.fh, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}; see {self.logpath}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3).read()
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"server did not become ready in {timeout}s; see {self.logpath}")

    def post(self, path, body, timeout=1800):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def get(self, path, timeout=60):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=timeout) as r:
            return json.load(r)

    def model_status(self, name):
        for m in self.get("/v1/models").get("data", []):
            if m.get("id") == name:
                st = m.get("status")
                return st.get("value") if isinstance(st, dict) else st
        return None

    def load_model(self, name, timeout=1800):
        """Explicitly load a model and return how long it took.

        The router is started with --no-models-autoload (as production does), so a completion
        against an unloaded model returns 400 "model is not loaded". Loading explicitly is also
        the only way to time the load cleanly - which is the metric that detects a regression in
        the tensor-split scatter path.
        """
        t0 = time.time()
        self.post("/models/load", {"model": name}, timeout=timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.model_status(name)
            if st == "loaded":
                return round(time.time() - t0, 1)
            if st in ("error", "failed"):
                raise RuntimeError(f"model {name} entered status '{st}'")
            time.sleep(2)
        raise TimeoutError(f"model {name} not loaded within {timeout}s (status={self.model_status(name)})")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        try:
            self.fh.close()
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()


# ---------------------------------------------------------------- measurement

def settle(max_wait=180, idle_gb=2.0):
    """Wait for the GPUs to actually be free before measuring.

    Learned the hard way: in the first gate run Devstral's candidate decode read 23.53 t/s against
    a 25.96 baseline - an apparent -9.4% regression. Re-measured with the two builds interleaved
    it was 26.55 vs 25.48, i.e. +4.2% the other way, with run-to-run spread under 0.02 t/s. The
    first number was contention from the previous model still releasing, not code. Any measurement
    taken while the previous one is still tearing down is worthless, so wait for it.
    """
    t0 = time.time()
    while time.time() - t0 < max_wait:
        used = vram_used_gb()
        if used is not None and used <= idle_gb:
            time.sleep(5)          # let clocks come down off the previous run too
            return True
        time.sleep(5)
    log(f"        (GPUs not idle after {max_wait}s; measurement may be contended)")
    return False


def vram_used_gb():
    """Total VRAM in use across the two discrete cards, in GiB."""
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--csv"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    total = 0.0
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].startswith("card") and parts[0] != "card2":
            try:
                total += int(parts[2]) / (1024 ** 3)
            except ValueError:
                pass
    return round(total, 2)


def completion(srv, model, prompt, n_predict=N_PREDICT, depth_filler=""):
    """One temp-0 completion. temp 0 + ignore_eos makes speculation lossless, so t/s from
    different spec settings is directly comparable and the output hash is a correctness anchor."""
    body = {
        "model": model,
        "prompt": depth_filler + prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
    }
    t0 = time.time()
    r = srv.post("/completion", body)
    wall = time.time() - t0
    tim = r.get("timings", {}) or {}
    return {
        "wall_s": round(wall, 3),
        "prefill_tps": tim.get("prompt_per_second"),
        "decode_tps": tim.get("predicted_per_second"),
        "draft_n": tim.get("draft_n"),
        "draft_n_accepted": tim.get("draft_n_accepted"),
        "sha256": hashlib.sha256((r.get("content") or "").encode()).hexdigest(),
    }


def acceptance(m):
    n, a = m.get("draft_n"), m.get("draft_n_accepted")
    if not n:
        return None
    return round(a / n, 4)


def filler_for_depth(tokens, varied=True):
    """Roughly `tokens` tokens of filler, to put the KV at a known depth.

    `varied` matters when the thing being compared is a SPECULATOR rather than a build. A
    repeating unit makes the model degenerate into repetition at temperature 0, and an n-gram
    speculator then predicts it almost perfectly - Muse-Glimmer measured 0.9922 acceptance on
    repeated filler, which says nothing about real traffic. Varied filler keeps acceptance in a
    meaningful range. Comparisons between two builds are unaffected either way, since both sides
    see the same text; comparisons between spec-types are not.
    """
    if tokens <= 0:
        return ""
    if not varied:
        return "The cache is warm. " * (tokens // 5) + "\n\n"

    # deterministic but non-repeating: same text every run, so results stay comparable
    rng = random.Random(1234)
    words = ("system memory bandwidth latency kernel tensor buffer gradient throughput cache "
             "scheduler pipeline quantise attention residual embedding checkpoint inference "
             "matrix register occupancy dispatch throughput reduction transpose").split()
    out, n = [], 0
    while n < tokens:
        sent = " ".join(rng.choice(words) for _ in range(rng.randint(8, 18)))
        out.append(sent.capitalize() + ".")
        n += len(sent.split()) + 1
    return " ".join(out) + "\n\n"


# ---------------------------------------------------------------- modes

def run_smoke(args):
    """Load one preset section at PRODUCTION context and generate. The point is not speed - it is
    that the heaviest, tightest-fitting configurations still load. A VRAM regression from the
    draft-KV-in-full patch shows up here as an OOM and nowhere else."""
    ini = pick_ini(args.presets, args.model)
    if ini is None:
        log(f"  model '{args.model}' not found in any preset under {args.presets}")
        return 1
    logpath = os.path.join(args.out, f"{safe(args.model)}.log")
    before = vram_used_gb()
    try:
        with Server(args.build, ini, args.port, logpath) as srv:
            load_s = srv.load_model(args.model)
            m = completion(srv, args.model, BASKET[0][1], n_predict=N_PREDICT)
            peak = vram_used_gb()
            rec = {
                "model": args.model, "ini": os.path.basename(ini),
                "load_seconds": load_s,
                "peak_vram_gb": None if peak is None else round(peak - (before or 0), 2),
                **m,
            }
            write_json(os.path.join(args.out, f"smoke-{safe(args.model)}.json"), rec)
            log(f"        load {load_s}s  prefill {m['prefill_tps']}  decode {m['decode_tps']}  "
                f"vram +{rec['peak_vram_gb']} GiB")
            return 0
    except Exception as e:
        log(f"        {type(e).__name__}: {e}")
        return 1


def run_gate(args):
    """Fast preset-faithful pass: one depth, full basket, through the server.

    llama-bench was removed from this harness deliberately. It cannot read a preset, so its flags
    had to be hand-derived from the ini - and that is exactly what went wrong: the derived set
    omitted nothing visible but silently ran f16 KV where the preset says q8_0, which inverted a
    decode result (+4.2% instead of -9.4%) and nearly buried a real regression. It also cannot do
    speculative decoding, which every model in this fleet uses. The server needs no translation
    layer: it reads the same ini production reads.
    """
    results = {}
    models = (args.models.split(",") if args.models else GATE_MODELS)
    for name in models:
        ini = pick_ini(args.presets, name)
        if ini is None:
            log(f"  skip {name}: not in presets")
            continue
        log(f"  gate {name}")
        settle()
        logpath = os.path.join(args.out, f"gate-{safe(name)}.log")
        try:
            before = vram_used_gb()
            with Server(args.build, ini, args.port, logpath) as srv:
                load_s = srv.load_model(name)
                peak = vram_used_gb()
                filler = filler_for_depth(16384)
                # discard the first run: clocks ramp and it lands ~10% low
                completion(srv, name, BASKET[0][1], n_predict=N_PREDICT, depth_filler=filler)
                per_cat = {}
                for cat, prompt in BASKET:
                    m = completion(srv, name, prompt, n_predict=N_PREDICT, depth_filler=filler)
                    per_cat.setdefault(cat, []).append(m)
                r = summarise(per_cat)
                r["load_seconds"] = load_s
                r["peak_vram_gb"] = None if peak is None else round(peak - (before or 0), 2)
                results[name] = r
                log(f"        prefill {r['prefill_tps']}  decode {r['decode_tps']}  "
                    f"accept {r['acceptance']}  load {load_s}s  vram +{r['peak_vram_gb']} GiB")
        except Exception as e:
            log(f"        {type(e).__name__}: {e}")
            results[name] = {"error": str(e)}
    return finish(args, "gate", results)


def run_deep(args):
    """Two-turn conversation per model, at the depth the preset is actually configured for.

    Fixed 8k/16k/32k depths test a configuration nobody runs. These presets are set up for
    c = 131072 and beyond, and the interesting failures live near that limit: VRAM fit at full
    context, checkpoint eviction, cache-reuse across turns, KV growth. So:

        turn 1  ->  ~50% of the model's configured context
        turn 2  ->  ~90% of it, continuing the SAME conversation with cache_prompt on

    Turn 2 reusing turn 1's KV is the realistic path and the only one that exercises
    cache-reuse / ctx-checkpoints the way production does.

    Capped by DEEP_CTX_CAP (default 131072) so the 262k and 700k presets do not spend an hour
    each in prefill; models configured below the cap are tested at their true maximum.
    """
    results = {}
    models = (args.models.split(",") if args.models else DEEP_MODELS)
    for name in models:
        ini = pick_ini(args.presets, name)
        if ini is None:
            continue
        ctx = preset_ctx(ini, name) or 32768
        # Room for the generation itself plus slack, or the last point overruns the context.
        headroom = N_PREDICT + 2048
        fixed = [d for d in DEPTHS if d + headroom <= ctx]
        half, full = ctx // 2, ctx - headroom
        logpath = os.path.join(args.out, f"deep-{safe(name)}.log")
        log(f"  {name}: configured ctx {ctx} -> single-shot {fixed}, then two-turn {half} -> {full}")
        settle()
        try:
            with Server(args.build, ini, args.port, logpath) as srv:
                srv.load_model(name)
                # warm-up must be long enough to actually ramp clocks, then is discarded
                completion(srv, name, BASKET[0][1], n_predict=N_PREDICT)

                # cheap points: single-shot, directly comparable across models
                for depth in fixed:
                    per_cat = {}
                    for cat, prompt in BASKET:
                        m = completion(srv, name, prompt, n_predict=N_PREDICT,
                                       depth_filler=filler_for_depth(depth))
                        per_cat.setdefault(cat, []).append(m)
                    results[f"{name}@{depth}"] = summarise(per_cat)
                    log(f"  {name}@{depth}: {results[f'{name}@{depth}']}")

                # deep points: one conversation, half-context then full-context, so turn 2 is a
                # cache hit on turn 1 - the only shape that exercises cache-reuse and the context
                # checkpoints at the depth these presets are actually configured for.
                # One prompt per category rather than all six: at 130k-258k tokens the depth is
                # what is being measured and the prompt barely moves the result, while each extra
                # repetition costs a full deep prefill. Keeping one code + one general preserves
                # the split that actually disagrees; the cheap points above keep all six.
                if full > max(fixed or [0]):
                    deep_basket = [next(x for x in BASKET if x[0] == "code"),
                                   next(x for x in BASKET if x[0] == "gen")]
                    for cat, prompt in deep_basket:
                        conv = two_turn(srv, name, prompt, half, full)
                        for turn, m in conv.items():
                            results.setdefault(f"{name}@{turn}", {"_per": []})["_per"].append((cat, m))
        except Exception as e:
            log(f"  {name}: {type(e).__name__}: {e}")
            results[f"{name}@error"] = {"error": str(e)}

    # collapse the per-prompt lists into medians
    for key, val in list(results.items()):
        if "_per" in val:
            per_cat = {}
            for cat, m in val["_per"]:
                per_cat.setdefault(cat, []).append(m)
            results[key] = summarise(per_cat)
            log(f"  {key}: {results[key]}")
    return finish(args, "deep", results)


def two_turn(srv, model, prompt, t1_tokens, t2_tokens):
    """One conversation, two turns, growing the KV toward the configured context limit.

    cache_prompt=True on turn 2 is the point: it makes the server reuse turn 1's KV instead of
    re-prefilling, which is how production runs and what exercises cache-reuse and the context
    checkpoints. Measuring turn 2 with a cold cache would test something else entirely.
    """
    out = {}

    turn1_prompt = filler_for_depth(t1_tokens) + prompt
    body1 = {"model": model, "prompt": turn1_prompt, "n_predict": N_PREDICT,
             "temperature": 0.0, "ignore_eos": True, "cache_prompt": True}
    t0 = time.time()
    r1 = srv.post("/completion", body1)
    out["turn1"] = shape(r1, time.time() - t0)

    # turn 2 continues the same text, so the shared prefix is a cache hit
    turn2_prompt = (turn1_prompt + (r1.get("content") or "")
                    + filler_for_depth(max(0, t2_tokens - t1_tokens - N_PREDICT))
                    + "\n\nNow revise the answer above and justify each change.")
    body2 = {"model": model, "prompt": turn2_prompt, "n_predict": N_PREDICT,
             "temperature": 0.0, "ignore_eos": True, "cache_prompt": True}
    t0 = time.time()
    r2 = srv.post("/completion", body2)
    out["turn2"] = shape(r2, time.time() - t0)
    return out


def shape(r, wall):
    tim = r.get("timings", {}) or {}
    return {
        "wall_s": round(wall, 3),
        "prefill_tps": tim.get("prompt_per_second"),
        "decode_tps": tim.get("predicted_per_second"),
        "draft_n": tim.get("draft_n"),
        "draft_n_accepted": tim.get("draft_n_accepted"),
        "n_prompt": tim.get("prompt_n"),
        "sha256": hashlib.sha256((r.get("content") or "").encode()).hexdigest(),
    }


def preset_ctx(ini, section):
    """Context available to ONE request, which is what a depth ladder must respect.

    `c` in the preset is the total context; with `np = N` slots and kv-unified = 0 it is split
    across them, so a single request only gets c/np. Ignoring this would drive the deep points
    past the slot limit on exactly the presets configured for the most context - e.g.
    Qwen3.6-35B-A3B:HQ (c = 524288, np = 2 -> 262144) and Qwen3.6-27B-Fable:FAST
    (c = 700000, np = 3 -> ~233333) - producing context shift or failure rather than a measurement.
    """
    merged = merged_keys(ini, section)
    ctx = None
    for k in ("c", "ctx-size"):
        if merged.get(k):
            try:
                ctx = int(merged[k])
                break
            except ValueError:
                pass
    if ctx is None:
        return None
    try:
        np_slots = max(1, int(merged.get("np", 1)))
    except ValueError:
        np_slots = 1
    return ctx // np_slots


def summarise(per_cat):
    flat = [m for lst in per_cat.values() for m in lst]
    def med(key):
        vals = sorted(v for v in (m.get(key) for m in flat) if v)
        return None if not vals else round(vals[len(vals) // 2], 2)
    accs = [a for a in (acceptance(m) for m in flat) if a is not None]
    return {
        "prefill_tps": med("prefill_tps"),
        "decode_tps": med("decode_tps"),
        "acceptance": None if not accs else round(sum(accs) / len(accs), 4),
        "sha_by_prompt": [m["sha256"][:12] for m in flat],
        "per_category": {c: round(sum(x["decode_tps"] for x in l if x["decode_tps"]) / max(1, len(l)), 2)
                         for c, l in per_cat.items()},
    }



# ---------------------------------------------------------------- sweep

def is_truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def all_sections(preset_dir):
    """Every model section across the preset files, minus [*] and the embedding/rerank entries.

    Embedding and rerank models are excluded by instruction: they do not generate, so decode,
    acceptance and a context sweep are all meaningless for them.
    """
    out = []
    for fn in sorted(os.listdir(preset_dir)):
        if not fn.endswith(".ini"):
            continue
        path = os.path.join(preset_dir, fn)
        txt = open(path, encoding="utf-8").read()
        for name in re.findall(r"^\[([^*\]][^\]]*)\]\s*$", txt, re.M):
            m = merged_keys(path, name)
            if is_truthy(m.get("embedding")) or is_truthy(m.get("reranking")):
                continue
            out.append((name, path))
    return out


def sweep_rungs(ctx):
    """8k, doubling, then the model's own ceiling.

    The final rung is 90% of the per-slot context rather than 100%: the generation needs room, and
    a run that trips context shift measures the shift rather than the model.
    """
    rungs, d = [], 8192
    ceiling = int(ctx * 0.90)
    while d < ceiling:
        rungs.append(d)
        d *= 2
    if not rungs or rungs[-1] < ceiling:
        rungs.append(ceiling)
    return [r for r in rungs if r > 0]


def run_sweep(args):
    """Every configuration in the presets, swept to its own maximum context."""
    todo = all_sections(args.presets)
    if args.models:
        want = set(args.models.split(","))
        todo = [t for t in todo if t[0] in want]

    # most informative first, so an interrupted run still answers the important questions
    priority = {n: i for i, n in enumerate(DEEP_MODELS)}
    todo.sort(key=lambda t: (priority.get(t[0], 99), t[0]))

    out_json = os.path.join(args.out, "sweep.json")
    results = load_json(out_json) or {}

    # Hard wall-clock budget. This sweep is 229 measurement points and keeps production stopped for
    # its whole duration, so it stops cleanly at the deadline rather than running until it happens
    # to finish. Results are written after every model, so a later window resumes where this left
    # off instead of repeating work.
    budget_s = int(os.environ.get("SWEEP_BUDGET_S", 8 * 3600))
    t_start = time.time()
    log("  %d configurations to sweep, budget %.1f h" % (len(todo), budget_s / 3600.0))

    for name, ini in todo:
        if name in results and results[name].get("rungs"):
            log("  skip %s (already recorded)" % name)
            continue
        elapsed = time.time() - t_start
        if elapsed > budget_s:
            log("")
            log("  BUDGET REACHED after %.1f h - stopping cleanly with %d/%d done; rerun to resume"
                % (elapsed / 3600.0, len(results), len(todo)))
            break
        ctx = preset_ctx(ini, name) or 32768
        rungs = sweep_rungs(ctx)
        entry = {"ini": os.path.basename(ini), "ctx_per_slot": ctx, "rungs": {}}
        log("")
        log("  %s  (ctx/slot %d) -> %s" % (name, ctx, rungs))
        settle()
        logp = os.path.join(args.out, "sweep-%s.log" % safe(name))
        try:
            before = vram_used_gb()
            with Server(args.build, ini, args.port, logp) as srv:
                entry["load_seconds"] = srv.load_model(name)
                peak = vram_used_gb()
                entry["vram_gb"] = None if peak is None else round(peak - (before or 0), 2)
                # one warm-up for the whole model, not per rung
                completion(srv, name, BASKET[0][1], n_predict=N_PREDICT)
                for depth in rungs:
                    # deep rungs cost a full prefill each: one prompt there, two when cheap
                    prompts = BASKET[:2] if depth <= 32768 else BASKET[:1]
                    per = {}
                    try:
                        for cat, prompt in prompts:
                            m = completion(srv, name, prompt, n_predict=N_PREDICT,
                                           depth_filler=filler_for_depth(depth))
                            per.setdefault(cat, []).append(m)
                        r = summarise(per)
                        entry["rungs"][str(depth)] = r
                        log("     %7d: prefill %-9s decode %-9s accept %s"
                            % (depth, r["prefill_tps"], r["decode_tps"], r["acceptance"]))
                    except Exception as e:
                        entry["rungs"][str(depth)] = {"error": "%s: %s" % (type(e).__name__, e)}
                        log("     %7d: FAILED %s: %s" % (depth, type(e).__name__, e))
                        break   # deeper rungs will fail the same way
        except Exception as e:
            entry["error"] = "%s: %s" % (type(e).__name__, e)
            log("     load/serve FAILED: %s: %s" % (type(e).__name__, e))
        results[name] = entry
        write_json(out_json, results)

    write_sweep_report(results, os.path.join(args.out, "sweep-report.md"))
    return 0


def write_sweep_report(results, path):
    ok = [n for n, e in results.items() if e.get("rungs") and not e.get("error")]
    bad = [n for n, e in results.items() if e.get("error")]
    lines = ["# Context sweep - every preset configuration", ""]
    lines.append("- configurations swept: %d" % len(results))
    lines.append("- loaded and generated: %d" % len(ok))
    lines.append("- failed to load or serve: %d" % len(bad))
    lines.append("")
    if bad:
        lines += ["## Failed", ""]
        for n in sorted(bad):
            lines.append("- **%s**: %s" % (n, results[n]["error"]))
        lines.append("")
    lines += ["## Results", "",
              "| model | ctx/slot | load s | VRAM GiB | depth | prefill t/s | decode t/s | accept |",
              "|---|---|---|---|---|---|---|---|"]
    for n in sorted(ok):
        e = results[n]
        for depth, r in sorted(e["rungs"].items(), key=lambda kv: int(kv[0])):
            if "error" in r:
                lines.append("| %s | %s | %s | %s | %s | - | - | FAILED |"
                             % (n, e["ctx_per_slot"], e.get("load_seconds"), e.get("vram_gb"), depth))
            else:
                lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |"
                             % (n, e["ctx_per_slot"], e.get("load_seconds"), e.get("vram_gb"),
                                depth, r["prefill_tps"], r["decode_tps"], r["acceptance"]))
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log("")
    log("  report written: %s" % path)


# ---------------------------------------------------------------- load timing

def drop_page_cache():
    """Model load time is meaningless unless the page cache state matches between runs.

    With mmap, a warm cache makes a 60 GB load look near-instant - comparing a cold baseline
    against a warm candidate would manufacture a huge fake 'improvement'. Only safe to call while
    production is stopped, which is the only time this harness runs.
    """
    try:
        subprocess.run(["sync"], timeout=120)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        time.sleep(2)
        return True
    except Exception as e:
        log(f"        (could not drop page cache: {e}; load times not comparable)")
        return False


def measure_load(build, ini, model, port, out, tag):
    """Time a real, cold model load through the server. This is the only trustworthy load-time
    number: it is measured on the model-load call itself, not inferred from a wall clock that
    also contains the generation."""
    settle()
    drop_page_cache()
    logpath = os.path.join(out, f"load-{tag}.log")
    try:
        with Server(build, ini, port, logpath) as srv:
            return srv.load_model(model)
    except Exception as e:
        log(f"        load timing failed: {type(e).__name__}: {e}")
        return None



# ---------------------------------------------------------------- preset helpers

def pick_ini(preset_dir, model):
    for fn in sorted(os.listdir(preset_dir)):
        if not fn.endswith(".ini"):
            continue
        path = os.path.join(preset_dir, fn)
        with open(path, encoding="utf-8") as f:
            if re.search(r"^\[%s\]\s*$" % re.escape(model), f.read(), re.M):
                return path
    return None


def merged_keys(ini, section):
    """Keys for one section with the [*] globals applied underneath, mirroring how the server
    cascades presets."""
    glob, sec, cur = {}, {}, None
    with open(ini, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith((";", "#")):
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line[1:-1]
                continue
            if "=" not in line:
                continue
            k, v = (x.strip() for x in line.split("=", 1))
            if cur == "*":
                glob[k] = v
            elif cur == section:
                sec[k] = v
    return {**glob, **sec}



# ---------------------------------------------------------------- reference

def finish(args, mode, results):
    meta = collect_meta(args)
    payload = {"meta": meta, "results": results}
    write_json(os.path.join(args.out, f"{mode}-results.json"), payload)

    if args.baseline:
        ref = load_json(args.ref) or {}
        ref.setdefault(mode, {})
        ref[mode] = {"meta": meta, "results": results}
        write_json(args.ref, ref)
        log(f"\n  recorded {len(results)} entries as the {mode} reference -> {args.ref}")
        return 0

    ref = load_json(args.ref)
    if not ref or mode not in ref:
        log(f"\n  no {mode} reference in {args.ref}; run once with --baseline against the "
            f"PRODUCTION build first")
        return 1
    return compare(ref[mode]["results"], results, os.path.join(args.out, "perf-report.txt"),
                   ref[mode].get("meta", {}), meta)


def compare(ref, cur, report_path, ref_meta, cur_meta):
    lines, bad = [], 0
    lines.append(f"reference build: {ref_meta.get('build')}  sha {ref_meta.get('sha')}")
    lines.append(f"candidate build: {cur_meta.get('build')}  sha {cur_meta.get('sha')}")
    lines.append("")
    lines.append(f"{'entry':<44} {'metric':<14} {'ref':>10} {'cur':>10} {'delta':>9}  verdict")
    for key in sorted(set(ref) | set(cur)):
        r, c = ref.get(key), cur.get(key)
        if r is None:
            lines.append(f"{key:<44} {'-':<14} {'-':>10} {'-':>10} {'-':>9}  NEW (no reference)")
            continue
        if c is None:
            lines.append(f"{key:<44} {'-':<14} {'-':>10} {'-':>10} {'-':>9}  MISSING in candidate")
            bad += 1
            continue
        for metric, tol in TOLERANCE.items():
            rv, cv = r.get(metric), c.get(metric)
            if rv in (None, 0) or cv is None:
                continue
            if metric == "acceptance":
                delta, ok = cv - rv, (cv - rv) >= tol
                shown = f"{delta:+.4f}"
            else:
                delta = (cv - rv) / rv
                ok = delta >= tol if metric in HIGHER_IS_BETTER else delta <= tol
                shown = f"{100*delta:+.1f}%"
            if not ok:
                bad += 1
            lines.append(f"{key:<44} {metric:<14} {rv:>10} {cv:>10} {shown:>9}  "
                         f"{'ok' if ok else 'REGRESSION'}")
        # output equivalence is a correctness signal, not perf - report separately
        if r.get("sha_by_prompt") and c.get("sha_by_prompt") and \
           r["sha_by_prompt"] != c["sha_by_prompt"]:
            lines.append(f"{key:<44} {'output-sha':<14} {'-':>10} {'-':>10} {'-':>9}  "
                         f"CHANGED (investigate before accepting)")
    text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(text + "\n")
    log("\n" + text)
    return 0 if bad == 0 else 1


def collect_meta(args):
    return {
        "build": args.build,
        "sha": read_first_line(os.path.join(os.path.dirname(args.build.rstrip("/")), "REBASE_SHA")),
        "rocm": run_capture(["hipconfig", "-R"]),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_clocks": run_capture(["bash", "-c",
                                   "rocm-smi --showclocks 2>/dev/null | head -20"]),
    }


def run_capture(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def read_first_line(path):
    try:
        with open(path) as f:
            return f.readline().strip()
    except Exception:
        return ""


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "gate", "deep", "sweep"])
    ap.add_argument("--build", required=True)
    ap.add_argument("--presets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="validation/reference.json")
    ap.add_argument("--model")
    ap.add_argument("--models")
    ap.add_argument("--port", type=int, default=20099)
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "smoke":
        if not args.model:
            ap.error("--model is required for smoke")
        return run_smoke(args)
    if args.mode == "gate":
        return run_gate(args)
    if args.mode == "sweep":
        return run_sweep(args)
    return run_deep(args)


if __name__ == "__main__":
    sys.exit(main())
