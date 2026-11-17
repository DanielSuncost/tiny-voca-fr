#!/usr/bin/env bash
# Overnight French run: corpus -> features -> alignment -> training -> evaluation.
# Every stage is resumable; rerunning the script continues where it stopped.
#   bash scripts/french/run_fr.sh data/french/voices.json fr_v1
set -euo pipefail
cd "$(dirname "$0")/../.."
VOICES=${1:?voices.json}
RUN=${2:-fr_v1}
export TINY_VOCA_LANGUAGE=fr PYTHONPATH=src
PY=.venv/bin/python
D=data/french
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "generate + screen"
$PY scripts/french/generate_fr.py --sentences $D/text/sentences.jsonl --voices "$VOICES" --output $D/dataset

stamp "acoustic features"
for split in train validation test; do
  $PY -m tiny_voca.prepare --inputs $D/dataset/$split.jsonl --split $split --output $D/features --resume
done

stamp "forced alignment"
REVISION="fr-$(git rev-parse --short HEAD)"
$PY -m tiny_voca.annotate --features $D/features --output $D/annotations --dataset-revision "$REVISION" --resume

stamp "training $RUN"
RESUME=()
[ -f runs/$RUN/last.pt ] && RESUME=(--resume runs/$RUN/last.pt)
$PY -m tiny_voca.train --train $D/annotations/train.jsonl --validation $D/annotations/validation.jsonl \
  --output runs/$RUN --steps 100000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 --max-seconds 86400 "${RESUME[@]}"

stamp "evaluation"
$PY -m tiny_voca.evaluate --checkpoint runs/$RUN/inference.pt --annotations $D/annotations --output eval/$RUN --tag $RUN
stamp "done"
