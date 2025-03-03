"""
This module provides a cli in
order to push outputs from `psrecord` to
`wandb`. It expects a `tmp.json` containing
the corresponding run group.
"""

import argparse
import json
import os
import yaml

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
    help="""
        List of output files seperated by semi-colon.
        Expects 2 files in the format `server_file;client_file`.
        """,
)
parser.add_argument(
    "--wandb_run_group", type=str, help="Run group for `wandb`", default=None
)
args = parser.parse_args()

output_files = args.output_files.split(";")
server_file = output_files[0]
client_file = output_files[1]

# Load settings
with open("settings.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Initialize wandb
run_group = args.wandb_run_group
if run_group is None:
    with open("tmp.json", "r", encoding="utf-8") as f:
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


def log_metrics(df, prefix, step):
    """Log CPU and memory metrics to wandb with a given prefix."""
    wandb.log(
        {
            f"{prefix}_cpu_percentage": df.loc[step, "CPU (%)"],
            f"{prefix}_real_memory_mb": df.loc[step, "Real Memory (MB)"],
            f"{prefix}_virtual_memory_mb": df.loc[step, "Virtual Memory (MB)"],
        },
        step=step,
    )


df_server_row_count = len(df_server)
df_client_row_count = len(df_client)

for i in range(max(df_server_row_count, df_client_row_count)):
    if i < df_server_row_count:
        log_metrics(df_server, "server", i)
    if i < df_client_row_count:
        log_metrics(df_client, "client", i)
