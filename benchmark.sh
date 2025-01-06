#!/bin/bash

# Function to display help text
show_help() {
    echo "Usage: $0 <model_type> <he_flag>"
    echo ""
    echo "Arguments:"
    echo "  <model_type>   The model type to use. Must be one of: fednn, fedqnn."
    echo "  <he_flag>      Enable or disable homomorphic encryption. Must be one of: true, false."
    echo ""
    echo "Description:"
    echo "  This script launches a Flower server and multiple clients for federated learning."
    echo "  The number of clients is specified in the settings.yaml file under 'number_clients'."
    echo ""
    echo "Examples:"
    echo "  $0 fednn true    # Run with 'fednn' model and homomorphic encryption enabled."
    echo "  $0 fedqnn false  # Run with 'fedqnn' model and homomorphic encryption disabled."
}

# Ensure yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: wireshark is not installed. Please install yq to continue."
    exit 1
fi

# Ensure tshark is installed
if ! command -v tshark &> /dev/null; then
    echo "Error: tshark is not installed. Please install tshark to continue."
    exit 1
fi

# Check if help is requested
if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Read the number of clients from settings.yaml
NUM_CLIENTS=$(yq '.number_clients' settings.yaml)

# Set the model type and FHE flag
MODEL_TYPE=$1
HE=$2

# Validate input
if [[ -z "$MODEL_TYPE" || -z "$HE" ]]; then
    echo "Error: Missing arguments."
    show_help
    exit 1
fi
if [[ "$MODEL_TYPE" != "fednn" && "$MODEL_TYPE" != "fedqnn" ]]; then
    echo "Error: Invalid model type. Please use 'fednn' or 'fedqnn'."
    show_help
    exit 1
fi
if [[ "$HE" != "true" && "$HE" != "false" ]]; then
    echo "Error: HE flag must be 'true' or 'false'."
    show_help
    exit 1
fi

# Set up wireshark
mkdir -p timelogs
tshark -q -z conv,tcp -f "tcp port 8150" -i any -B 30 > timelogs/tshark_output.txt &
tshark_pid=$!
tshark -i any -T fields -e frame.len -f "tcp port 8150" -B 30 > timelogs/tshark_packets.txt &
tshark_pid2=$!

# Start the Flower server
echo "Starting Flower server..."
python server.py --he "$HE" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

server_time_logs="timelogs/flwr_server_PID${SERVER_PID}.txt"
psrecord $SERVER_PID --log $server_time_logs --interval 0.5 &

# Allow server to initialize
sleep 5

# Start Flower clients
for ((i = 0; i < NUM_CLIENTS; i++)); do
    echo "Starting client $i with model $MODEL_TYPE..."
    python client.py --client_index "$i" --he "$HE" --model "$MODEL_TYPE" &
    CLIENT_PIDS[$i]=$!
    echo "New Client PID: ${CLIENT_PIDS[$i]}"
    client_time_logs="timelogs/flwr_client${i}_PID${CLIENT_PIDS[$i]}.txt"
    psrecord ${CLIENT_PIDS[$i]} --log $client_time_logs --interval 0.5 &
done

# Wait for processes to complete
wait $SERVER_PID
for PID in "${CLIENT_PIDS[@]}"; do
    wait $PID
done

kill $tshark_pid
kill $tshark_pid2

echo "Training completed."
