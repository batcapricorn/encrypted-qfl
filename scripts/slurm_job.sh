#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --partition=clara
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

module load Anaconda3/2024.02-1
source activate base
pip install flwr==1.5.0 tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas opacus pyyaml tenseal psrecord yq wandb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if [ ! -d "/dev/shm/data" ]; then
        echo "Copying dataset to /dev/shm/..."
        cp -r ../data/ /dev/shm/
else
        echo "Dataset already exists in /dev/shm/, skipping copy."
fi

./scripts/experiment.sh $MODEL_TYPE $HE_FLAG