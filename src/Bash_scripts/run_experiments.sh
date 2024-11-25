#!/bin/bash
#### ie: bash src/Bash_scripts/run_experiments.sh data_config/ACDC_256.yaml 1 10 0 test


# Function to handle SIGINT signal
interrupt_handler() {
    echo "Interrupted. Exiting..."
    exit 1
}

# Assign SIGINT signal to interrupt_handler function
trap interrupt_handler SIGINT


# Assign input arguments to variables
MODEL_CONFIG="model_config/medsam_config.yaml"
MODULE_CONFIG="model_config/promptmodule_config.yaml"
TRAIN_CONFIG="train_config/train_config_100_0001.yaml"

DATA_CONFIG=$1 # ie: "data_config/data_config_ACDC_256.yaml"
CLASS_TO_SEGMENT=$2  # 1 or 3
NUM_SAMPLES=$3 # 0 or 10
GPU_IDX=$4
LOGGERNAME=$5

# Logging input parameters for debugging
echo "Input Parameters: "
echo "Model Config: $MODEL_CONFIG"
echo "Model Config: $MODULE_CONFIG"
echo "Data Config: $DATA_CONFIG"
echo "Class to Segment: $CLASS_TO_SEGMENT"
echo "Train Config: $TRAIN_CONFIG"
echo "Logger Name: $LOGGERNAME"

# Define the base command
BASE_CMD="python src/main.py"

if [ $NUM_SAMPLES -eq 0 ]; then
    declare -a train_indice_list=("") 
elif [ $NUM_SAMPLES -eq 10 ]; then
    case $DATA_CONFIG in
        data_config/ACDC_256.yaml)
            declare -a train_indice_list=("287 280 473 320 213 761 535 662 678 123" "619 596 483 557 479 722 212 125 729 243" "91 386 394 38 407 138 757 30 480 334")
            ;;
        data_config/CAMUS_512.yaml)
            declare -a train_indice_list=("36 348 292 252 98 334 217 70 103 8" "133 163 68 314 279 141 288 209 195 92" "108 249 5 153 74 78 334 171 259 329")
            ;;
        data_config/HC_640.yaml)
            declare -a train_indice_list=("387 152 248 107 236 8 491 33 398 440" "21 506 3 418 422 143 188 9 374 68" "207 407 65 299 200 206 181 504 338 326")
            ;;
        *)
            echo "Unknown DATASET configuration: $DATA_CONFIG"
            exit 1
            ;;
    esac
else
    echo "NUM_SAMPLES must be 0 or 10."
    exit 1
fi

# Define arrays of options for each variable parameter
declare -a loss_configs=("loss_config/outerBCE_W1.yaml" "loss_config/tightbox_W00001.yaml" "loss_config/boxsize_w001.yaml")
declare -a positional_encoding=("fixed")
declare -a seeds=("0" "1" "2")

# Loop through each combination of parameters
for seed in "${seeds[@]}"; do
  for pos_encoding in "${positional_encoding[@]}"; do
    for loss in "${loss_configs[@]}"; do
      # Start constructing the command
      CMD="$BASE_CMD --data_config $DATA_CONFIG --prompt_config prompt_config/box_tight.yaml --data__class_to_segment $CLASS_TO_SEGMENT --sam_config $MODEL_CONFIG --module_config $MODULE_CONFIG --train_config $TRAIN_CONFIG --loss_config $loss --data__use_precomputed_sam_embeddings --gpu_idx $GPU_IDX --train__sam_positional_encoding $pos_encoding --seed $seed"                    
      
      # Conditionally add --train__train_indices if it's not empty
      if [ ! -z "$train_indices" ]; then
        CMD="$CMD --train__train_indices $train_indices"
      fi
      
      echo "Executing: $CMD"
      eval $CMD || {
        echo "Error executing command: $CMD"
        # Optionally, log error details or take additional actions here
        continue
      }
    done
  done
done