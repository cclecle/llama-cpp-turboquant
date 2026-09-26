# scripts/fleet - the tools used to validate and tune the fleet on the box

These ship inside the release tarball and are run from `/opt/llamacpp/llama-cpp-mine-vN/scripts/fleet/`.
They are here so they are versioned and visible; nothing this workflow depends on should live in
`/tmp` or `/root`. One-off measurement scripts go under `/root/work-<date>/` and are deleted or
committed when the work ends.

| tool | what it does |
|---|---|
| `build-release.sh <N>` | configure + build a release on the box with the production options |
| `fleetcheck.sh` / `fleetcheck.py` | pre-cutover smoke: one mid rung per family on both stores, 2k prompt + 512 gen, VRAM, text sanity; stops and restarts production |
| `rungpass.sh <main.ini> <tag> <ids...>` | measure rungs of a SINGLEGPU family through a scratch router whose args mirror llamacpp-0 |
| `rungpass-dual.sh <main.ini> <tag> <ids...>` | same for DUALGPU (GPU 0+1, tensor split), args mirror llamacpp-both |
| `rungmeasure.py <port> <ids...>` | the per-rung measurement: load via /models/load, 10k-token prefill + decode, VRAM after generation (peak proxy, max over cards), text sanity, unload |
| `visiontest.py <port> <id> <image> [max_tokens]` | ask a real image question on a VISION rung; shows reasoning vs visible answer |
| `rungargs.py <unit> <rung> <out>` | the exact child args the router last used for a rung (from the unit journal), one per line, ready for `mapfile` |
| `rungab.sh <args file> <devices> <chars> <release>...` | one rung's exact args on several builds, same /root/ppl.txt prompt, greedy: prompt_n, pp/tg and the answer head per build |
| `benchab.sh <tag> <devices> <relA> <relB> -- <llama-bench args>` | same llama-bench run on two builds (ENV_A / ENV_B for per-arm env), one line per build |
| `tbo-diff.sh <relA> <relB> [op]` | full test-backend-ops for two builds in parallel (A on GPU 0, B on GPU 1) and the reachable-failure diff |
| `ctxcheck.py [release]` | rungs whose per-slot context exceeds what the model can use (pre-YaRN original ctx when a YaRN model runs with rope-scaling none); GPU-less scratch routers, safe while production runs |
| `ggufkv.py <file.gguf> [substring...]` | GGUF metadata reader without numpy |
| `rungsweep.sh <release> [frac]` | EVERY rung of both stores once (resumable): prompt = frac (0.5) of the per-slot context, 256 greedy tokens, load time, pp/tg, VRAM after generation, gibberish check; logs `sweep-<release>-{single,dual}.log` |
| `multi-prompt.sh <args-file> <devices> <vA> <vB>` | same rung, 6 prompts, two releases, pooled tg. A single greedy prompt is a trajectory sample, not a measurement |

Conventions the tools assume: production units `llamacpp-0` (GPU 0, `SINGLEGPU/main.ini`), `llamacpp-1`
(GPU 1), `llamacpp-both` (GPU 0+1, `DUALGPU/main.ini`); scratch router port 20099; `/root/ppl.txt` as the
prompt corpus (any few-MB C++ text); `BIN=<path to llama-server>` overrides the release the
`fleetcheck`/`rungpass` scripts run (default: the production release when they were last updated); the 34.2 GB cards with a ship target of 33.5-33.9 GB used at peak.
The router prints the exact child command line after `spawning server instance with args:` in its log -
that is the args file `multi-prompt.sh` takes, and the faithful way to run a rung standalone.
