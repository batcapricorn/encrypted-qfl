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
    help="List of output files seperated by semi-colon. Expects 2 files in the format `server_file;client_file`.",
)
parser.add_argument(
    "--wandb_run_group", type=str, help="Run group for `wandb`", default=None
)
args = parser.parse_args()

output_files = args.output_files.split(";")
server_file = output_files[0]
client_file = output_files[1]

# Load settings
with open("settings.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize wandb
run_group = args.wandb_run_group
if run_group is None:
    with open("tmp.json", "r") as f:
        wandb_config = json.load(f)

    run_group = wandb_config.get("WANDB_RUN_GROUP")

wandb.init(
    project=config["wandb_project"],
    config={
        "group": run_group,
    },
    name="system_logs",
)

df_server = pd.read_csv(
    os.path.join(args.output_dir, server_file),
    sep=r"\s+",
    skiprows=1,
    names=[
        "Elapsed Time (s)",
        "CPU (%)",
        "Real Memory (MB)",
        "Virtual Memory (MB)",
    ],
)
df_client = pd.read_csv(
    os.path.join(args.output_dir, client_file),
    sep=r"\s+",
    skiprows=1,
    names=[
        "Elapsed Time (s)",
        "CPU (%)",
        "Real Memory (MB)",
        "Virtual Memory (MB)",
    ],
)

df_server_row_count = len(df_server)
df_client_row_count = len(df_client)
for i in range(max(df_server_row_count, df_client_row_count)):
    if i + 1 <= df_server_row_count:
        wandb.log(
            {
                f"server_cpu_percentage": df_server.loc[i, "CPU (%)"],
                f"server_real_memory_mb": df_server.loc[i, "Real Memory (MB)"],
                f"server_virtual_memory_mb": df_server.loc[i, "Virtual Memory (MB)"],
            },
            step=i,
        )
    if i + 1 <= df_client_row_count:
        wandb.log(
            {
                f"client_cpu_percentage": df_client.loc[i, "CPU (%)"],
                f"client_real_memory_mb": df_client.loc[i, "Real Memory (MB)"],
                f"client_virtual_memory_mb": df_client.loc[i, "Virtual Memory (MB)"],
            }
        )
