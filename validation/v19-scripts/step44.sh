#!/bin/bash
# step44.sh : v19, the dev tree back at HEAD (the step 43 kernel removed): build, one load, the text must be the
# reference 15c95895521e9ee4fc6aa67dc23f6ff1 at ~45.3 ms/step. Production stays stopped.
cd /mnt/gguf/r9v/bench
T=/opt/llamacpp/tmp-v19
systemctl stop llamacpp-0 llamacpp-1 llamacpp-both; sleep 3
echo "### build"
(cd $T && bash validation/v17-scripts/build-scratch.sh $T 2>&1 | head -6)
FN_ARGS=fn-xl-tiered-r9vlike.args FN_TAG=-s44 bash flashnext.sh v19dev 2>&1 | grep -E 'load:|died'
python3 step_ms.py v19dev-s44 results.jsonl
md5sum v19dev-s44.answer.txt
echo STEP44_DONE
