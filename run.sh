#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# General configuration
stage=1              # Processes starts from the specified stage
stop_stage=100       # Processes is stopped at the specified stage
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages

# Data preparation related
data_dir=             # Input data directory

# Training related
train_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# Pretrained model related
pretrained_asr=       # Pretrained ASR model
pretrained_lm=        # Pretrained LM model

# Decoding related
use_lm=true           # Use language model for decoding
use_wordlm=false      # Use word-based language model
nbest=1               # Output N-best hypotheses
beam_size=20          # Beam size
penalty=0.0           # Insertion penalty
maxlenratio=0.0       # Maximum length ratio in decoding
minlenratio=0.0       # Minimum length ratio in decoding
ctc_weight=0.5        # CTC weight in joint decoding
lm_weight=0.3         # Language model weight in decoding

. ./path.sh
. ./cmd.sh

# Parse command-line options
. utils/parse_options.sh || exit 1

# Check required arguments
if [ -z "${data_dir}" ]; then
    log "Error: --data_dir is required"
    exit 1
fi

# Data preparation stage
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && ! ${skip_data_prep}; then
    log "Stage 1: Data preparation"
    
    # Run data preparation script
    python3 local/data_prep.py --data_dir "${data_dir}"
    
    # Check if data directories exist
    for split in train valid test; do
        if [ ! -d data/${split} ]; then
            log "Error: data/${split} directory does not exist"
            exit 1
        fi
    done
fi

# Feature extraction stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && ! ${skip_data_prep}; then
    log "Stage 2: Feature extraction"
    
    # Extract features for each split
    for split in train valid test; do
        utils/fix_data_dir.sh data/${split}
        
        # Compute global CMVN
        compute-cmvn-stats --binary=false scp:data/${split}/feats.scp \
            data/${split}/cmvn.ark
    done
fi

# Language model training stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && ! ${skip_train}; then
    log "Stage 3: Language model training"
    
    # Train language model
    ${cuda_cmd} --gpu 1 exp/lm/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu 1 \
        --backend pytorch \
        --verbose 1 \
        --train-label data/train/text \
        --valid-label data/valid/text \
        --output exp/lm
fi

# ASR model training stage
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && ! ${skip_train}; then
    log "Stage 4: ASR model training"
    
    # Train ASR model
    ${cuda_cmd} --gpu 1 exp/asr/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu 1 \
        --backend pytorch \
        --outdir exp/asr \
        --tensorboard-dir tensorboard/asr \
        --debugmode 1 \
        --dict data/lang/dict.txt \
        --debugdir exp/asr \
        --minibatches 0 \
        --verbose 1 \
        --resume "" \
        --train-json data/train/data.json \
        --valid-json data/valid/data.json
fi

# Decoding stage
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && ! ${skip_eval}; then
    log "Stage 5: Decoding"
    
    # Decode with ASR model
    ${cuda_cmd} --gpu 1 exp/asr/decode/decode.log \
        asr_recog.py \
        --config ${inference_config} \
        --ngpu 1 \
        --backend pytorch \
        --batchsize 0 \
        --recog-json data/test/data.json \
        --result-label exp/asr/decode/result.json \
        --model exp/asr/model.acc.best
fi

# Scoring stage
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ] && ! ${skip_eval}; then
    log "Stage 6: Scoring"
    
    # Score the results
    sclite \
        -r exp/asr/decode/result.wrd.trn trn \
        -h exp/asr/decode/result.wrd.hyp trn \
        -i rm -o all stdout > exp/asr/decode/result.wrd.scoring
    
    log "WER: $(grep -oP 'Sum/Avg.*\|\s+\K[0-9]+\.[0-9]+' exp/asr/decode/result.wrd.scoring)%"
fi

log "Run completed"
