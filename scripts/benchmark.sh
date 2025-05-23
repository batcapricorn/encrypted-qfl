#!/bin/bash

# Function to display help text
show_help() {
    echo "Usage: $0 <model_type> <he_flag>"
    echo ""
    echo "Arguments:"
    echo "  <model_type>   The model type to use, e.g., fednn, fedqnn or fedqcnn."
    echo "  <he_flag>      Enable or disable homomorphic encryption."
    echo ""
    echo "Description:"
    echo "  This script launches a Flower server and multiple clients for federated learning."
    echo "  The number of clients is specified in the settings.yaml file under 'number_clients'."
    echo ""
    echo "Examples:"
    echo "  $0 fednn --he    # Run with 'fednn' model and homomorphic encryption enabled."
    echo "  $0 fedqnn  # Run with 'fedqnn' model and homomorphic encryption disabled."
}

# Check if help is requested
if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Read the number of clients from settings.yaml
NUM_CLIENTS=$(grep 'number_clients:' settings.yaml | cut -d':' -f2 | xargs)

# Set the model type and FHE flag
MODEL_TYPE=$1
shift
HE_FLAG=""

if [[ "$1" == "--he" ]]; then
    HE_FLAG="--he"
fi

# Validate input
if [[ -z "$MODEL_TYPE" ]]; then
    echo "Error: Missing arguments."
    show_help
    exit 1
fi
if [[ "$MODEL_TYPE" != "fednn" && "$MODEL_TYPE" != "fedqnn" && "$MODEL_TYPE" != "qcnn" && "$MODEL_TYPE" != "resnet18" && "$MODEL_TYPE" != "resnet18-qnn" && "$MODEL_TYPE" != "resnet18-qcnn" ]]; then
    echo "Error: Invalid model type. Please use 'fednn', 'fedqnn', 'qcnn', 'resnet18', 'resnet18-qnn' or 'resnet18-qcnn'."
    show_help
    exit 1
fi

# Create run group name for `wandb`
if [ -n "$SLURM_JOB_ID" ]; then
    ID="$SLURM_JOB_ID"
else
    ID=$(uuidgen | cut -c1-8)
fi

HE_NAME="Standard"
if [ "$HE_FLAG" == "--he" ]; then
    HE_NAME="FHE"
fi

WANDB_RUN_GROUP="${HE_NAME}-${MODEL_TYPE}-$ID"
export WANDB_RUN_GROUP
echo "WANDB_RUN_GROUP set to: $WANDB_RUN_GROUP"

# Start the Flower server
echo "Starting Flower server..."
python src/server.py --model "$MODEL_TYPE" --wandb_run_group "$WANDB_RUN_GROUP" $HE_FLAG &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Set up psrecords
mkdir -p timelogs
server_time_logs="timelogs/flwr_server_PID${SERVER_PID}.txt"
psrecord $SERVER_PID --log $server_time_logs --interval 0.5 &

# Allow server to initialize
sleep 60

# Start Flower clients
for ((i = 0; i < NUM_CLIENTS; i++)); do
    echo "Starting client $i with model $MODEL_TYPE..."
    python src/client.py --client_index "$i" --model "$MODEL_TYPE" --wandb_run_group "$WANDB_RUN_GROUP" $HE_FLAG &
    CLIENT_PIDS[$i]=$!
    echo "New Client PID: ${CLIENT_PIDS[$i]}"

    if [[ $i -eq 0 ]]; then
        # Only run psrecord for the first client
        client_time_logs="timelogs/flwr_client${i}_PID${CLIENT_PIDS[$i]}.txt"
        psrecord ${CLIENT_PIDS[$i]} --log $client_time_logs --interval 0.5 &
    fi
done


# Wait for processes to complete
wait $SERVER_PID
for PID in "${CLIENT_PIDS[@]}"; do
    wait $PID
done

# Push psrecord output to wandb
python3 src/scripts/psrecord_to_wandb.py --output_files="flwr_server_PID${SERVER_PID}.txt;flwr_client0_PID${CLIENT_PIDS[0]}.txt" --wandb_run_group="$WANDB_RUN_GROUP"

echo "Training completed."
