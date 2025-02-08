"""
This module provides a cli in
order to push outputs from `psrecord` to
`wandb`. It expects a `tmp.json` containing
the corresponding run group.
"""

import argparse
import yaml
import json
import os

import pandas as pd
import wandb

parser = argparse.ArgumentParser(
    prog="psrecords to wandb", description="Tool to push psrecord outputs to wandb"
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Directory that contains output files",
    default="timelogs/",
)

parser.add_argument(
    "--output_files",
    type=str,
    help="List of output files seperated by semi-colon",
)
args = parser.parse_args()

output_files = args.output_files.split(";")

# Load settings
with open("settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize wandb
with open("tmp.json", "r") as f:
    wandb_config = json.load(f)

run_group = wandb_config.get("WANDB_RUN_GROUP")
print(f"Run group: {run_group}")

wandb.init(
    project=config["wandb_project"],
    config={
        "group": run_group,
    },
    name="system_logs",
)

for file in output_files:
    df = pd.read_csv(
        os.path.join(args.output_dir, file),
        sep=r"\s+",
        skiprows=1,
        names=[
            "Elapsed Time (s)",
            "CPU (%)",
            "Real Memory (MB)",
            "Virtual Memory (MB)",
        ],
    )
    participant = "server" if "server" in file else "client"
    for index, row in df.iterrows():
        wandb.log(
            {
                f"{participant}_cpu_percentage": row["CPU (%)"],
                f"{participant}_real_memory_mb": row["Real Memory (MB)"],
                f"{participant}_virtual_memory_mb": row["Virtual Memory (MB)"],
            },
            step=index,
        )
