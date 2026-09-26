#!/bin/bash
# series 4: adaptive MTP caps 5/6 vs fixed 5 on 27B; AllReduce modes on 2-GPU tensor split; production stopped by caller
cd /root/work-20260925
export MP_NO_RESTART=1
for cap in 5 6; do
python3 - $cap <<'PY'
import sys
a=open('a27.args').read().split('\n')
i=a.index('--spec-type'); a[i+1]=a[i+1].replace('draft-mtp,','draft-mtp-adaptive,')
j=a.index('--spec-draft-n-max'); a[j+1]=sys.argv[1]
open('a27ad%s.args'%sys.argv[1],'w').write('\n'.join(a))
PY
echo "##### adaptive cap $cap vs fixed 5"; MP_ARGS_B=a27ad$cap.args bash multi-prompt.sh a27.args 0 rdnab3 rdnab3b 2>&1 | grep MEAN
grep -E 'failed|error' mp-rdnab3b.log | head -3
done
B=/opt/llamacpp/llama-cpp-mine-rdnab3/build3/bin
M=/mnt/gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q8_0.gguf
for E in 'X=0' 'GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_HIP_P2P=1' 'GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_HIP_P2P=0' 'GGML_CUDA_ALLREDUCE=internal GGML_CUDA_AR_HIP_P2P=2'; do
  echo "##### AR $E"
  env $E HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=$B timeout 1200 $B/llama-bench -m $M -sm tensor -ngl 999 -fa 1 -t 12 -ub 1536 -b 4096 -p 4096 -n 64 -r 3 2>&1 | grep -E 'pp4096|tg64|rror'
done
echo SERIES_DONE
