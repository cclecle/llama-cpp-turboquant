#!/bin/bash
# step15.sh : v19, the host side of the decode passes.
#   - GGML_CUDA_STAGED_UPLOAD (default on): small writes into compute buffers (the graph inputs, ~20 per pass per
#     device, each a copy + a stream synchronize of ~22 us while the GPU idled: step 14's HIP trace) go through a
#     pinned ring on an upload stream, ordered on the GPU with the graphs (no host wait)
#   - GGML_CUDA_MMVF_F32_TUNE (default on): F32 mat-vecs, 4 rows per block when there are many rows (the MoE router,
#     bit-identical) and wider blocks when there are fewer rows than CUs (the hyper-connection mixers)
# Decode is deterministic since step 14 (the one-kernel top-k), so correctness is checked on the text: A and C must
# generate step 14's B text byte for byte (the staging changes no math); D (wider blocks round differently) must give
# the same text on both loads. Production stays stopped.
#   1) test-backend-ops MUL_MAT (f32), 2) throughput A C D D C A + text md5, 3) trace of D: anatomy + host gaps
cd /mnt/gguf/r9v/bench
B=/opt/llamacpp/llama-cpp-mine-v19dev/build3/bin
ENV_A="GGML_CUDA_STAGED_UPLOAD=0 GGML_CUDA_MMVF_F32_TUNE=0"
ENV_C="GGML_CUDA_MMVF_F32_TUNE=0"
ENV_D="X=0"
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3

echo "### test-backend-ops"
echo "MUL_MAT f32: $(HIP_VISIBLE_DEVICES=0 timeout 1800 $B/test-backend-ops -o MUL_MAT -b ROCm0 -p type_a=f32 2>&1 | grep -E 'tests passed|FAIL' | head -6 | tr '\n' ' ')"

echo "### throughput, warm, A C D D C A"
n=0
for arm in A C D D C A; do
  n=$((n + 1)); eval E=\$ENV_$arm
  FN_ENV="$E" FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s15-$arm$n bash flashnext.sh v19dev 2>&1 | grep -E 'load:'
done
python3 step_ms.py v19dev-s15- results.jsonl
echo "### generated text per load (md5; step 14 B: f5225a17419d673cefad625c3538fb4b)"
md5sum v19dev-s15-*.answer.txt

echo "### kernel + HIP trace: D"
DP_HIP=1 DP_ARGS=fn-xl-tiered-r9vlike.args DP_PROMPT=prompt-32k.txt DP_TOKENS=512 DP_TAG=-s15 bash decode-profile.sh v19dev 2>&1 | grep -E 'timings|trace:'
d=prof-decode-v19dev-s15
f=$(find $d -name '*kernel_trace.csv' | head -1)
steps=$(python3 -c "import json; d=json.load(open('$d/answer.json'))['timings']; print(d['predicted_n'] - d['draft_n_accepted'])")
python3 torchtrace_anatomy.py rocprof $f $steps | head -8
python3 mmv_breakdown.py $f $steps "" mul_mat_vec_f | head -8
python3 hostgap_anatomy.py $d $steps
echo STEP15_DONE
