#!/bin/bash

# Parameters
# You can modify these lists or override variables
APPLIANCES=("fridge" "microwave" "kettle" "dishwasher" "washingmachine")
ITERATION=5000
SEQ_LEN=128
BATCH_SIZE=256

# Stop on error is not default in bash, but we can check status codes manually or use set -e
# set -e 

# Colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}   TimeGAN Automation: Train & Sample (Linux/WSL)${NC}"
echo -e "${CYAN}====================================================${NC}"
echo "Appliances: ${APPLIANCES[*]}"
echo "Iterations: $ITERATION"
echo "Sequence Length: $SEQ_LEN"
echo "Batch Size: $BATCH_SIZE"
echo -e "${CYAN}====================================================${NC}"

# Ensure we are in the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

for app in "${APPLIANCES[@]}"; do
    echo -e "\n${YELLOW}>>> Processing Appliance: [${app^^}]${NC}"
    
    # Construct command string for display
    CMD="python main_timegan.py --data_name $app --iteration $ITERATION --seq_len $SEQ_LEN --batch_size $BATCH_SIZE"
    
    echo -e "${GRAY}Running: $CMD${NC}"
    
    # Run Python command
    python main_timegan.py \
        --data_name "$app" \
        --iteration "$ITERATION" \
        --seq_len "$SEQ_LEN" \
        --batch_size "$BATCH_SIZE"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}TimeGAN failed for $app with exit code $EXIT_CODE${NC}"
        # Provide option to continue or exit? The PS1 script effectively continues but prints error.
        # But PS1 had $ErrorActionPreference = "Stop", so it actually stops.
        # Let's break here to match "Stop" behavior if that was intended, 
        # normally loop scripts continue. PS1 `Write-Error` is non-terminating unless specific preference.
        # PS1 line 9: $ErrorActionPreference = "Stop". So it stops on error.
        exit $EXIT_CODE
    fi
    
    # Verify output
    EXPECTED_OUTPUT="results/${app}_synthetic_data.npy"
    if [ -f "$EXPECTED_OUTPUT" ]; then
        echo -e "${CYAN}Successfully generated: $EXPECTED_OUTPUT${NC}"
    else
        echo -e "${YELLOW}WARNING: Output file not found at expected location: $EXPECTED_OUTPUT${NC}"
    fi
    
done

echo -e "\n${CYAN}====================================================${NC}"
echo -e "${CYAN}   All TimeGAN tasks completed!${NC}"
echo -e "${CYAN}====================================================${NC}"
