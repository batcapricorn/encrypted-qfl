#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=barnard
#SBATCH --time=2-00:00:00
#SBATCH --job-name=fl_simulation      # Job name
#SBATCH --mem=200G                     # Total memory
#SBATCH --output=fl_simulation_%j.out # Standard output
#SBATCH --error=fl_simulation_%j.err  # Standard error

MODEL_TYPE=$1
shift
HE_FLAG=""

if [[ "$1" == "--he" ]]; then
    HE_FLAG="--he"
fi

module load release/24.10
module load Anaconda3/2024.02-1
source .venv/bin/activate
#pip install flwr==1.5.0 tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas opacus pyyaml tenseal psrecord yq wandb
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if [ ! -d "/tmp/data" ]; then
        echo "Copying dataset to /tmp..."
        cp -r ../data/ /tmp
else
        echo "Dataset already exists in /tmp, skipping copy."
fi

wandb login

./scripts/benchmark.sh $MODEL_TYPE $HE_FLAG