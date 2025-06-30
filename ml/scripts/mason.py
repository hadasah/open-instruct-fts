import argparse
import re
import sys
from typing import List, Dict
import beaker
import os
import secrets
import string
from rich.console import Console
from rich.text import Text
import select
import time
import random

console = Console()


# ----------------------------------------------------------------------
# Open Instruct logic
OPEN_INSTRUCT_COMMANDS = [
    "open_instruct/finetune.py",
    "open_instruct/dpo_tune_cache.py",
    "open_instruct/grpo_fast.py",
    "open_instruct/ppo_fast.py",
    "open_instruct/grpo_vllm_thread_ray_gtrl.py",
    "open_instruct/ppo2.py",
    "open_instruct/ppo_vllm_thread_ray_gtrl.py",
    "open_instruct/reward_modeling.py",
]

OPEN_INSTRUCT_RESUMABLES = [
    "open_instruct/grpo_fast.py",
]

# ----------------------------------------------------------------------
# Mason logic
def parse_beaker_dataset(dataset_str):
    splt = dataset_str.split(":")
    if len(splt) != 2:
        raise argparse.ArgumentError()

    return {"mount_path": splt[0], "beaker": splt[1]}


def parse_env_var(env_var_str: str) -> Dict[str, str]:
    """Parse environment variable string in the format 'name=value'"""
    if '=' not in env_var_str:
        raise argparse.ArgumentTypeError(
            f"Environment variable must be in format 'name=value', got: {env_var_str}"
        )
    name, value = env_var_str.split('=', 1)
    if not name:
        raise argparse.ArgumentTypeError("Environment variable name cannot be empty")
    return {"name": name, "value": value}



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        type=str,
        nargs="+",
        help="Beaker hostname on which the job could be run.",
        default=None
    )
    parser.add_argument("--max_retries", type=int, help="Number of retries", default=0)
    parser.add_argument("--budget", type=str, help="Budget to use.", required=True)
    parser.add_argument("--gpus", type=int, help="Number of gpus", default=0)
    parser.add_argument("--num_nodes", type=int, help="Number of nodes", default=1)
    parser.add_argument(
        "--image",
        type=str,
        help="Beaker base image; usually fine to use AI2 base image.",
        default="ai2/cuda11.8-cudnn8-dev-ubuntu20.04",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="The Beaker workspace to use. If not set, use your default.",
        default=None,
    )
    parser.add_argument(
        "--beaker_datasets",
        nargs="*",
        help="""Beaker datasets to mount. You may give more than one, separated by
        spaces. Each dataset should be formatted like `[mount-point]:[beaker-dataset-id]`;
        for instance `/models:01HQXGAYGCS6D4ZK51K83CM49Y`.
        """,
        type=parse_beaker_dataset,
        default=[],
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Optionally, a description for this job in Beaker.",
        default="Beaker-Mason job.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name for the Beaker task.",
        default="beaker_mason"
    )
    parser.add_argument(
        "--priority", type=str, help="Beaker job priority.", default="normal"
    )
    parser.add_argument(
        "--preemptible", action="store_true", help="If given, run as preemptible"
    )
    parser.add_argument(
        "--pure_docker_mode", action="store_true", help="If given, run in pure docker mode"
    )
    parser.add_argument(
        "--no_hf_cache_env", action="store_true", help="Getting deprecated; it does nothing"
    )
    parser.add_argument(
        "--no_mount_nfs", action="store_true", help="Getting deprecated; it does nothing"
    )
    parser.add_argument(
        "--resumable", action="store_true", help="If given, make the job resumable"
    )
    parser.add_argument(
        "--no_auto_dataset_cache", action="store_true", help="If given, don't cache the dataset automatically"
    )
    parser.add_argument(
        "--auto_output_dir_path", type=str, default="/weka/oe-adapt-default/allennlp/deletable_checkpoint",
        help="If given, automatically replace the `--output_dir` argument with this path, essentially using it as a prefix"
    )
    parser.add_argument(
        "--auto_checkpoint_state_dir", type=str, default="/weka/oe-adapt-default/allennlp/deletable_checkpoint_states",
        help="If given, automatically replace the `--checkpoint_state_dir` argument with this path, essentially using it as a prefix"
    )
    parser.add_argument(
        "--env",
        type=parse_env_var,
        action="append",
        help="""Additional environment variables in the format 'name=value'. 
        Can be specified multiple times. Example: --env MY_VAR=value1 --env OTHER_VAR=value2""",
        default=[],
    )
    parser.add_argument(
        "--secret",
       type=parse_env_var,
        action="append",
        help="""Additional secret env variables in the format 'name=value'.
        Can be specified multiple times. Example: --secret MY_VAR=value1 --secret OTHER_VAR=value2""",
        default=[],
    )
    parser.add_argument(
        "--no-host-networking",
        action="store_true",
        help="If set, don't use host networking in experiment. Note this will make multi-node jobs error.",
    )
    # Split up the mason args from the Python args.
    mason_args, command_args = parser.parse_known_args()
    commands = parse_commands(command_args)
    return mason_args, commands



def parse_commands(command_args: List[str]) -> List[List[str]]:
    """the inputs are ['--', 'which', 'python', '--', 'echo', 'hello'], and this function converts it into [['which', 'python'], ['echo', 'hello']]"""
    if command_args[0] != "--":
        msg = (
            "Please separate the Python command you want to run with ' -- ', like "
            "`mason [mason-args] -- python [python-args]`."
        )
        raise Exception(msg)
    
    commands = []
    command = []
    for item in command_args:
        if item == "--":
            if command:
                commands.append(command)
                command = []
        else:
            command.append(item)
    if command:
        commands.append(command)
    return commands


    # useful_secrets = [
    #     "HF_TOKEN",
    #     "WANDB_API_KEY",
    #     "BEAKER_TOKEN",
    #     "OPENAI_API_KEY",
    #     # litellm expects these env vars
    #     "AZURE_API_KEY",
    #     "AZURE_API_BASE",
    #     "ANTHROPIC_API_KEY",
    # ]
    # if resumable:
    #     env_vars.extend([
    #         beaker.EnvVar(
    #             name="WANDB_RUN_ID",
    #             value=global_wandb_id,
    #         ),
    #         beaker.EnvVar(
    #             name="WANDB_RESUME",
    #             value="allow",
    #         ),
    #     ])

def make_internal_command(command: List[str], args: argparse.Namespace, whoami: str, is_external_user: bool) -> str:
    # pass through WANDB_ENTITY and WANDB_PROJECT
    if "WANDB_ENTITY" in os.environ:
        command = [f"WANDB_ENTITY={os.environ['WANDB_ENTITY']}"] + command
    if "WANDB_PROJECT" in os.environ:
        command = [f"WANDB_PROJECT={os.environ['WANDB_PROJECT']}"] + command
    if "WANDB_TAGS" in os.environ:
        command = [f"WANDB_TAGS={os.environ['WANDB_TAGS']}"] + command
    
    # escape the command (e.g., --stop_strings "</answer>")
    for i in range(len(command)):
        if "</" in command[i]:
            command[i] = f"'{command[i]}'"
    # breakpoint()

    is_open_instruct_training = any(cmd in command for cmd in OPEN_INSTRUCT_COMMANDS)
    if is_open_instruct_training:
        from open_instruct.dataset_transformation import get_commit_hash
        from open_instruct.utils import download_from_hf, gs_folder_exists, upload_to_gs_bucket
        # HACK: Cache dataset logic:
        # Here we basically try to run the tokenization full_command locally before running it on beaker
        # We could in theory submit a cpu only job to beaker to do this, but that requires setting up
        # dependency jobs somehow. Since tokenization is like ~5 minutes, we can just run it locally.
        # Once it's cached, we don't need to cache it again.
        def find_list_idx(lst: List[str], item: str):
            for i in range(len(lst)):
                if item == lst[i]:
                    return i
            return -1

        def remove_arg_from_list(lst: List[str], item: str, remove_value: bool = False):
            idx = find_list_idx(lst, item)
            if idx != -1 and idx + 1 < len(lst):
                if remove_value:
                    lst.pop(idx + 1)
                lst.pop(idx)

        # Add the whoami parts if not already present
        if not any("hf_entity" in c for c in command):
            command.append("--hf_entity")
            command.append("allenai")
        if not any("wandb_entity" in c for c in command):
            command.append("--wandb_entity")
            command.append("ai2-llm")
        
        dataset_cache_paths = []
        dataset_config_hashes = []
        if not args.no_auto_dataset_cache:
            for file in OPEN_INSTRUCT_COMMANDS:
                # add cache_dataset_only to the command
                idx = find_list_idx(command, file)
                if idx != -1:
                    # then try executing the same command with 
                    caching_command = command.copy()
                    remove_arg_from_list(caching_command, "--with_tracking", False)
                    remove_arg_from_list(caching_command, "--checkpoint_state_freq", True)
                    remove_arg_from_list(caching_command, "--checkpoint_state_dir", True)
                    remove_arg_from_list(caching_command, "--gs_checkpoint_state_dir", True)
                    caching_command = "python " + " ".join(caching_command[idx:]) + " --cache_dataset_only"
                    console.log(f"📦📦📦 Running the caching command with `--cache_dataset_only`")
                    import subprocess
                    # Use Popen to get real-time output while also capturing it
                    process = subprocess.Popen(
                        caching_command, 
                        shell=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )
                    
                    stdout_data, stderr_data = [], []
                    
                    # Set up select to monitor both stdout and stderr
                    streams = [process.stdout, process.stderr]
                    while True:
                        # Wait for output on either stream
                        reads = select.select(streams, [], [])[0]
                        
                        done = True
                        for stream in reads:
                            line = stream.readline()
                            if line:
                                done = False
                                is_stdout = stream == process.stdout
                                print(line.rstrip(), file=sys.stdout if is_stdout else sys.stderr)
                                if is_stdout:
                                    stdout_data.append(line)
                                else:
                                    stderr_data.append(line)
                        
                        if done and process.poll() is not None:
                            break
                            
                    result = type('SubprocessResult', (), {
                        'returncode': process.returncode,
                        'stdout': ''.join(stdout_data),
                        'stderr': ''.join(stderr_data)
                    })
                    stdout = result.stdout

            commit_hash = get_commit_hash(model_name_or_path, model_revision, "config.json", "model")
            download_from_hf(model_name_or_path, model_revision) # first download the model
            path = download_from_hf(model_name_or_path, model_revision) # then get the path
            gs_saved_path = f"gs://ai2-llm/post-training/deletable_cache_models/{model_name_or_path}/{commit_hash}"
            gs_folder = gs_folder_exists(gs_saved_path) # race condition exists, but it's fine since we are launching mason sequentially
            if not gs_folder:
                upload_to_gs_bucket(path, gs_saved_path)

            download_path = gs_saved_path.replace("gs://", "/gs/")
            download_path_without_last_folder = download_path.rsplit("/", 1)[0]
            gs_download_command = [
                "mkdir", "-p", download_path,
                "&&",
                "gsutil",
                "-o", f"GSUtil:parallel_thread_count=1",
                "-o", f"GSUtil:sliced_object_download_threshold=150",
                "-m",
                "cp", "-r", gs_saved_path, download_path_without_last_folder,
                "&&", "ls", download_path_without_last_folder,
                "&&", "ls", download_path,
                "&&",
            ]
            # Save dataset to GCS
            if len(dataset_cache_paths) > 0:
                for cidx, (dataset_cache_path, dataset_config_hash) in enumerate(zip(dataset_cache_paths, dataset_config_hashes)):
                    gs_saved_path = f"gs://ai2-llm/post-training/deletable_cache_datasets/{dataset_cache_path}"
                    gs_folder = gs_folder_exists(gs_saved_path) # race condition exists, but it's fine since we are launching mason sequentially
                    if not gs_folder:
                        upload_to_gs_bucket(dataset_cache_path, gs_saved_path)
                    dataset_cache_path_without_last_folder = dataset_cache_path.rsplit("/", 1)[0]
                    gs_download_command += [
                        "mkdir", "-p", dataset_cache_path_without_last_folder,
                        "&&",
                        "gsutil",
                        "cp", "-r", gs_saved_path, dataset_cache_path_without_last_folder,
                        "&&", "ls", dataset_cache_path_without_last_folder,
                        "&&", "ls", dataset_cache_path,
                        "&&",
                    ]
                    if cidx == 0:
                        command.append("--dataset_config_hash")
                        command.append(dataset_config_hash)
                    elif cidx == 1:
                        command.append("--dataset_config_eval_hash")
                        command.append(dataset_config_hash)
            command = gs_download_command + command

    return full_command

def main():
    args, commands = get_args()

    full_commands = [make_internal_command(command, args, whoami, is_external_user) for command in commands]

    for idx, full_command in enumerate(full_commands):
        console.rule(f"[bold blue]Command {idx+1}[/bold blue]")
        console.print(Text(full_command))

    experiment_spec = beaker.ExperimentSpec(
        description=args.description,
        tasks=[make_task_spec(args, full_command, i, beaker_secrets, whoami, args.resumable) for i, full_command in enumerate(full_commands)],
        budget=args.budget,
        retry=beaker.RetrySpec(allowed_task_retries=args.max_retries)
    )
    exp = beaker_client.experiment.create(spec=experiment_spec)

if __name__ == "__main__":
    main()
