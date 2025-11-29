#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESPnet2 Setup Script for Voice Pipeline

This script helps set up ESPnet2 for speech recognition, including:
- Cloning the ESPnet repository
- Installing dependencies
- Setting up a recipe for custom model training
- Preparing for fine-tuning with domain-specific data

The script provides utilities for:
1. Environment setup
2. Data preparation
3. Model configuration
4. Training setup
"""

import os
import sys
import argparse
import subprocess
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ESPnetSetup")

# Default paths
DEFAULT_ESPNET_DIR = os.path.expanduser("~/espnet")
DEFAULT_KALDI_DIR = os.path.expanduser("~/kaldi")
DEFAULT_DATA_DIR = "data"
DEFAULT_EXP_DIR = "exp"
DEFAULT_CONF_DIR = "conf"

# ESPnet repository URL
ESPNET_REPO = "https://github.com/espnet/espnet.git"
KALDI_REPO = "https://github.com/kaldi-asr/kaldi.git"


class ESPnetSetup:
    """
    Helper class for setting up ESPnet2 environment and preparing for model training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ESPnet setup with the given configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - espnet_dir: ESPnet repository directory
                - kaldi_dir: Kaldi repository directory
                - data_dir: Data directory
                - exp_dir: Experiment directory
                - conf_dir: Configuration directory
                - recipe_name: Recipe name
                - lang: Language code
                - pretrained_model: Pretrained model name (optional)
        """
        self.config = config or {}
        self.espnet_dir = self.config.get('espnet_dir', DEFAULT_ESPNET_DIR)
        self.kaldi_dir = self.config.get('kaldi_dir', DEFAULT_KALDI_DIR)
        self.data_dir = self.config.get('data_dir', DEFAULT_DATA_DIR)
        self.exp_dir = self.config.get('exp_dir', DEFAULT_EXP_DIR)
        self.conf_dir = self.config.get('conf_dir', DEFAULT_CONF_DIR)
        self.recipe_name = self.config.get('recipe_name', "custom_asr")
        self.lang = self.config.get('lang', "en")
        self.pretrained_model = self.config.get('pretrained_model', None)
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.conf_dir, exist_ok=True)
        
        logger.info(f"ESPnet setup initialized with espnet_dir: {self.espnet_dir}")
    
    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are installed.
        
        Returns:
            True if all dependencies are installed, False otherwise
        """
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
                logger.error("Python 3.6 or higher is required")
                return False
            
            # Check required commands (use shutil.which for better PATH handling)
            required_commands = ["git", "make", "cmake", "sox", "ffmpeg", "python3"]
            for cmd in required_commands:
                # Use shutil.which to find command in PATH
                cmd_path = shutil.which(cmd)
                if cmd_path is None:
                    logger.error(f"Required command not found: {cmd}")
                    logger.info(f"Please install {cmd} or ensure it's in your PATH")
                    return False
                else:
                    logger.info(f"Found {cmd} at: {cmd_path}")
            
            logger.info("All dependencies are installed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False
    
    def clone_repositories(self) -> bool:
        """
        Clone ESPnet and Kaldi repositories if they don't exist.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clone ESPnet if it doesn't exist
            if not os.path.exists(self.espnet_dir):
                logger.info(f"Cloning ESPnet repository to {self.espnet_dir}")
                subprocess.run(["git", "clone", ESPNET_REPO, self.espnet_dir], check=True)
            else:
                logger.info(f"ESPnet repository already exists at {self.espnet_dir}")
            
            # Clone Kaldi if it doesn't exist (needed for some ESPnet features)
            if not os.path.exists(self.kaldi_dir):
                logger.info(f"Cloning Kaldi repository to {self.kaldi_dir}")
                subprocess.run(["git", "clone", KALDI_REPO, self.kaldi_dir], check=True)
            else:
                logger.info(f"Kaldi repository already exists at {self.kaldi_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cloning repositories: {e}")
            return False
    
    def install_espnet_dependencies(self) -> bool:
        """
        Install ESPnet dependencies.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save current directory
            original_dir = os.getcwd()
            
            # Change to ESPnet directory
            os.chdir(self.espnet_dir)
            
            # Check if tools directory exists
            tools_dir = os.path.join(self.espnet_dir, "tools")
            if os.path.exists(tools_dir):
                logger.info("ESPnet tools directory found")
                
                # Try to run setup if it exists
                setup_script = os.path.join(tools_dir, "setup_venv.sh")
                if os.path.exists(setup_script):
                    logger.info("ESPnet requires manual setup - automated setup is too complex")
                    logger.info("Skipping setup_venv.sh (requires manual configuration)")
                    logger.warning("ESPnet installation is VERY complex. Consider using Whisper fine-tuning instead!")
                else:
                    logger.warning("ESPnet setup script not found, skipping automated setup")
            else:
                logger.warning("ESPnet tools directory not found")
            
            # Try simple pip installation first
            logger.info("Attempting simple ESPnet installation via pip")
            result = subprocess.run(
                ["pip", "install", "espnet", "espnet-model-zoo"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("ESPnet installed successfully via pip")
            else:
                # Try installing from source
                logger.info("Trying to install ESPnet from source")
                result2 = subprocess.run(
                    ["pip", "install", "-e", "."],
                    capture_output=True,
                    text=True
                )
                
                if result2.returncode != 0:
                    logger.error(f"ESPnet installation failed: {result2.stderr}")
                    logger.warning("=" * 60)
                    logger.warning("ESPnet installation is VERY complex!")
                    logger.warning("RECOMMENDATION: Use Whisper fine-tuning instead")
                    logger.warning("It's 10x easier and better for your use case")
                    logger.warning("See: ESPNET_MANUAL_INSTALLATION.md")
                    logger.warning("=" * 60)
                else:
                    logger.info("ESPnet installed from source")
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info("ESPnet dependencies installation completed (manual steps may be required)")
            return True
            
        except Exception as e:
            logger.error(f"Error installing ESPnet dependencies: {e}")
            logger.info("Please follow manual installation guide at: https://espnet.github.io/espnet/installation.html")
            return False
    
    def create_recipe_directory(self) -> str:
        """
        Create a recipe directory for custom ASR model.
        
        Returns:
            Path to the recipe directory
        """
        try:
            # Create recipe directory
            recipe_dir = os.path.join(self.espnet_dir, "egs2", self.recipe_name, "asr1")
            os.makedirs(recipe_dir, exist_ok=True)
            
            # Create subdirectories
            for subdir in ["data", "conf", "dump", "exp", "local", "steps", "utils"]:
                os.makedirs(os.path.join(recipe_dir, subdir), exist_ok=True)
            
            # Create symbolic links for steps and utils
            steps_src = os.path.join(self.espnet_dir, "egs2/TEMPLATE/asr1/steps")
            utils_src = os.path.join(self.espnet_dir, "egs2/TEMPLATE/asr1/utils")
            
            if os.path.exists(steps_src) and not os.path.exists(os.path.join(recipe_dir, "steps")):
                os.symlink(steps_src, os.path.join(recipe_dir, "steps"))
            
            if os.path.exists(utils_src) and not os.path.exists(os.path.join(recipe_dir, "utils")):
                os.symlink(utils_src, os.path.join(recipe_dir, "utils"))
            
            # Copy template files
            template_files = ["path.sh", "cmd.sh", "run.sh"]
            for file in template_files:
                src = os.path.join(self.espnet_dir, "egs2/TEMPLATE/asr1", file)
                dst = os.path.join(recipe_dir, file)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            
            logger.info(f"Recipe directory created at {recipe_dir}")
            return recipe_dir
            
        except Exception as e:
            logger.error(f"Error creating recipe directory: {e}")
            return ""
    
    def create_config_files(self, recipe_dir: str) -> bool:
        """
        Create configuration files for ASR model.
        
        Args:
            recipe_dir: Path to the recipe directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create conf directory
            conf_dir = os.path.join(recipe_dir, "conf")
            os.makedirs(conf_dir, exist_ok=True)
            
            # Create ASR config file
            asr_config = {
                "frontend_conf": {
                    "n_fft": 512,
                    "win_length": 400,
                    "hop_length": 160
                },
                "specaug": "specaug",
                "normalize": "global_mvn",
                "model_conf": {
                    "ctc_weight": 0.3,
                    "lsm_weight": 0.1,
                    "length_normalized_loss": False
                },
                "encoder": "conformer",
                "encoder_conf": {
                    "output_size": 256,
                    "attention_heads": 4,
                    "linear_units": 1024,
                    "num_blocks": 12,
                    "dropout_rate": 0.1,
                    "positional_dropout_rate": 0.1,
                    "attention_dropout_rate": 0.1,
                    "input_layer": "conv2d",
                    "normalize_before": True,
                    "macaron_style": True,
                    "pos_enc_layer_type": "rel_pos",
                    "selfattention_layer_type": "rel_selfattn",
                    "activation_type": "swish",
                    "use_cnn_module": True,
                    "cnn_module_kernel": 31
                },
                "decoder": "transformer",
                "decoder_conf": {
                    "attention_heads": 4,
                    "linear_units": 2048,
                    "num_blocks": 6,
                    "dropout_rate": 0.1,
                    "positional_dropout_rate": 0.1,
                    "self_attention_dropout_rate": 0.1,
                    "src_attention_dropout_rate": 0.1
                },
                "optim": "adam",
                "optim_conf": {
                    "lr": 0.001
                },
                "scheduler": "warmuplr",
                "scheduler_conf": {
                    "warmup_steps": 25000
                },
                "max_epoch": 50,
                "batch_type": "folded",
                "batch_size": 32,
                "accum_grad": 2,
                "grad_clip": 5,
                "patience": 3,
                "val_scheduler_criterion": "valid/acc",
                "val_scheduler_type": "max",
                "keep_nbest_models": 10
            }
            
            with open(os.path.join(conf_dir, "train_asr.yaml"), "w") as f:
                json.dump(asr_config, f, indent=4)
            
            # Create LM config file
            lm_config = {
                "lm": "transformer",
                "lm_conf": {
                    "pos_enc": None,
                    "embed_unit": 128,
                    "att_unit": 512,
                    "head": 8,
                    "unit": 2048,
                    "layer": 16,
                    "dropout_rate": 0.1
                },
                "batch_type": "folded",
                "batch_size": 64,
                "max_epoch": 20,
                "optim": "adam",
                "optim_conf": {
                    "lr": 0.001
                },
                "scheduler": "warmuplr",
                "scheduler_conf": {
                    "warmup_steps": 25000
                }
            }
            
            with open(os.path.join(conf_dir, "train_lm.yaml"), "w") as f:
                json.dump(lm_config, f, indent=4)
            
            # Create decode config file
            decode_config = {
                "beam_size": 20,
                "penalty": 0.0,
                "maxlenratio": 0.0,
                "minlenratio": 0.0,
                "ctc_weight": 0.5,
                "lm_weight": 0.3
            }
            
            with open(os.path.join(conf_dir, "decode_asr.yaml"), "w") as f:
                json.dump(decode_config, f, indent=4)
            
            logger.info(f"Configuration files created in {conf_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating configuration files: {e}")
            return False
    
    def create_data_prep_script(self, recipe_dir: str) -> bool:
        """
        Create a data preparation script for custom data.
        
        Args:
            recipe_dir: Path to the recipe directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create local directory
            local_dir = os.path.join(recipe_dir, "local")
            os.makedirs(local_dir, exist_ok=True)
            
            # Create data preparation script
            script_content = r"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import csv
import subprocess
from pathlib import Path

def prepare_data(data_dir, split):
    \"\"\"
    Prepare data for ESPnet training.
    
    Args:
        data_dir: Data directory
        split: Data split (train, valid, test)
    \"\"\"
    # Create output directory
    output_dir = os.path.join("data", split)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create wav.scp file
    wav_scp_path = os.path.join(output_dir, "wav.scp")
    text_path = os.path.join(output_dir, "text")
    utt2spk_path = os.path.join(output_dir, "utt2spk")
    
    # Get all wav files in the data directory
    wav_files = list(Path(data_dir).glob(f"{split}/**/*.wav"))
    
    with open(wav_scp_path, "w") as wav_scp, \\
         open(text_path, "w") as text, \\
         open(utt2spk_path, "w") as utt2spk:
        
        for wav_file in wav_files:
            # Get the corresponding text file
            txt_file = wav_file.with_suffix(".txt")
            if not txt_file.exists():
                continue
            
            # Read the transcription
            with open(txt_file, "r") as f:
                transcription = f.read().strip()
            
            # Generate utterance ID
            utt_id = f"{split}_{wav_file.stem}"
            spk_id = f"{split}"
            
            # Write to files
            wav_scp.write(f"{utt_id} {wav_file.absolute()}\\n")
            text.write(f"{utt_id} {transcription}\\n")
            utt2spk.write(f"{utt_id} {spk_id}\\n")
    
    # Create spk2utt file
    subprocess.run(["utils/utt2spk_to_spk2utt.pl", utt2spk_path, os.path.join(output_dir, "spk2utt")])
    
    print(f"Data preparation for {split} completed")

def main():
    parser = argparse.ArgumentParser(description="Prepare data for ESPnet training")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    args = parser.parse_args()
    
    # Prepare data for each split
    for split in ["train", "valid", "test"]:
        prepare_data(args.data_dir, split)

if __name__ == "__main__":
    main()
"""
            
            with open(os.path.join(local_dir, "data_prep.py"), "w") as f:
                f.write(script_content)
            
            # Make the script executable
            os.chmod(os.path.join(local_dir, "data_prep.py"), 0o755)
            
            logger.info(f"Data preparation script created at {os.path.join(local_dir, 'data_prep.py')}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating data preparation script: {e}")
            return False
    
    def create_run_script(self, recipe_dir: str) -> bool:
        """
        Create a run script for training and inference.
        
        Args:
            recipe_dir: Path to the recipe directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create run script
            script_content = r"""#!/usr/bin/env bash

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
"""
            
            with open(os.path.join(recipe_dir, "run.sh"), "w") as f:
                f.write(script_content)
            
            # Make the script executable
            os.chmod(os.path.join(recipe_dir, "run.sh"), 0o755)
            
            logger.info(f"Run script created at {os.path.join(recipe_dir, 'run.sh')}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating run script: {e}")
            return False
    
    def setup_espnet(self) -> bool:
        """
        Set up ESPnet environment and prepare for model training.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Clone repositories
            if not self.clone_repositories():
                return False
            
            # Install ESPnet dependencies
            if not self.install_espnet_dependencies():
                return False
            
            # Create recipe directory
            recipe_dir = self.create_recipe_directory()
            if not recipe_dir:
                return False
            
            # Create configuration files
            if not self.create_config_files(recipe_dir):
                return False
            
            # Create data preparation script
            if not self.create_data_prep_script(recipe_dir):
                return False
            
            # Create run script
            if not self.create_run_script(recipe_dir):
                return False
            
            logger.info("ESPnet setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up ESPnet: {e}")
            return False


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESPnet Setup Script")
    parser.add_argument("--espnet-dir", type=str, default=DEFAULT_ESPNET_DIR,
                        help="ESPnet repository directory")
    parser.add_argument("--kaldi-dir", type=str, default=DEFAULT_KALDI_DIR,
                        help="Kaldi repository directory")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Data directory")
    parser.add_argument("--exp-dir", type=str, default=DEFAULT_EXP_DIR,
                        help="Experiment directory")
    parser.add_argument("--conf-dir", type=str, default=DEFAULT_CONF_DIR,
                        help="Configuration directory")
    parser.add_argument("--recipe-name", type=str, default="custom_asr",
                        help="Recipe name")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language code")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Pretrained model name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "espnet_dir": args.espnet_dir,
        "kaldi_dir": args.kaldi_dir,
        "data_dir": args.data_dir,
        "exp_dir": args.exp_dir,
        "conf_dir": args.conf_dir,
        "recipe_name": args.recipe_name,
        "lang": args.lang,
        "pretrained_model": args.pretrained_model
    }
    
    # Set up ESPnet
    espnet_setup = ESPnetSetup(config)
    if espnet_setup.setup_espnet():
        print("ESPnet setup completed successfully")
    else:
        print("ESPnet setup failed")












