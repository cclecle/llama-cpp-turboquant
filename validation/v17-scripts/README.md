# v17 validation runners (2026-09-25/26)

One-off orchestration used for the v17 validation, kept so every measurement can be re-run. The reusable tools
they call live in `scripts/fleet/`. All ran on the box from `/root/work-2026092{5,6}/`, with production stopped.

| script | what it did |
|---|---|
| `build-scratch.sh` | configure + build a scratch tree with the production cmake options (`/opt/llamacpp/tmp-rdnab`) |
| `tbo.sh`, `tbofull.sh` | test-backend-ops v16 vs new (per op, then full) - now `scripts/fleet/tbo-diff.sh` |
| `getargs.py` | router child args from the journal - now `scripts/fleet/rungargs.py` |
| `series.sh` .. `series4.sh` | the multi-prompt A/B series (short/deep context, band on/off, adaptive MTP, AllReduce modes) |
| `regress.sh` | fork-regress on v16 and the new build (run it on a FULL build tree, see the study) |
| `repro.sh` | the Mistral-Small-4:S `-nckvc` + `-sm tensor` crash reproduction (35k prompt past the host boundary) |
| `rungab.sh` | first version of `scripts/fleet/rungab.sh` |
| `hd512.sh` | head 320/512/576 WMMA A/B (rejected: tile wins on gemma-4) |
| `r9.sh` | the r9 typed-store fix A/B across families + the r5 f16 band settings |
| `series5.sh` | band retune under MTP (v17 vs attn, 6 prompts, short and deep) + the tensor-split ppl "abort" line |
| `attn.sh` | derived kq mask / native q8_0 prefill / band retune: FA tests, compute buffers, perplexity equality, prefill and decode A/B (`/opt/llamacpp/tmp-attn`) |
