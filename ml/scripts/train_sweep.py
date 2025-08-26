import argparse
import itertools
import json
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import (
    DD_MODEL_NAME_TEMPLATE,
    DD_REVISION_TEMPLATE,
    DD_MODEL_SIZES,
    DD_MODEL_SIZES_INFO,
    DD_TRAIN_SETS,
    DD_MODEL_SEEDS,
    OPEN_INSTRUCT_COMMANDS,
    MODEL_HP_DEFAULTS,
    COMMAND_HP_DEFAULTS,
    PROJECT_SPECS,
    HARDWARE_SPECS_DICT,
)
from utils import seq_dict_update


def main(
    sweep_name,
    models=None,
    model_type=None,
    model_train_data_sets=None,
    model_sizes=None,
    model_revisions=None,
    model_seeds=None,
    model_steps=None,
    command=None,
    relaunch_path=None,
    relaunch_name=None,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    job_time='24:00:00',
    gpus=None,
    cpus=None,
    mem=None,
    include_jobs_indices=None,
    ignore_specs_check_keys=["NUM_CPUS", "MEM_GB"],
    filter_succeeded=True,
    filter_running=True,
    use_local_model=True,
    **kwargs,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")
    if command is None:
        raise RuntimeError("Must specify command to run for each job")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    job_time = '1:00:00' if debug else job_time
    user = os.environ.get('USER')
    if user not in PROJECT_SPECS:
        raise ValueError(f"User {user} not found in PROJECT_SPECS. Please add your user to the PROJECT_SPECS dictionary.")
    USER_SPECS = PROJECT_SPECS[user]

    command_prefix = f"{USER_SPECS['PROJECT_DIR']}/open_instruct/{command}.py"

    if relaunch_path or relaunch_name:
        if relaunch_name and relaunch_path:
            raise ValueError("Cannot specify both relaunch_name and relaunch_path")
        if relaunch_name:
            relaunch_path = os.path.join(PROJECT_SPECS[user]['DEFAULT_SAVE_PATH'], relaunch_name)

        relaunch_path = relaunch_path.rstrip('/')
        model_sweep_name = os.path.basename(relaunch_path)
        path_to_grid_file = os.path.join(relaunch_path, 'grid.json')
        path_to_specs = os.path.join(relaunch_path, 'specs.json')
        if not os.path.exists(path_to_grid_file):
            raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
        grid = json.load(open(path_to_grid_file, 'r'))
        model = grid.get('main_grid', {}).get('model_name', [None])[0]

        SPECS = seq_dict_update([USER_SPECS, HARDWARE_SPECS_DICT['all'], HARDWARE_SPECS_DICT.get(model, {}).get(partition)])
        SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
        SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
        SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

        if os.path.exists(path_to_specs):
            old_specs = json.load(open(path_to_specs, 'r'))
            for key in old_specs:
                if key not in ignore_specs_check_keys:
                    assert SPECS.get(key) == old_specs[key], f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"
        
        default_grid = seq_dict_update([
            MODEL_HP_DEFAULTS.get('all'), MODEL_HP_DEFAULTS.get(model, {}), COMMAND_HP_DEFAULTS.get("all"), COMMAND_HP_DEFAULTS.get(command, {})
        ])

        run_grid(
            grid,
            default_grid=default_grid,
            sweep_name=model_sweep_name,
            specs=SPECS,
            name_keys=SPECS.get("NAME_KEYS", []),
            prefix=command_prefix,
            gpus=SPECS['NUM_GPUS'],
            cpus=SPECS["NUM_CPUS"],
            nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
            node_exclude=None,
            account=account,
            partition=partition,
            DIR_PATH=SPECS["PROJECT_DIR"],
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),            
            include_job_id=False,
            hashname=False,
            replace_jobname_slashes=True,
            sweep_wandb_tags=[command], 
            saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
            mem_gb=SPECS["MEM_GB"],
            requeue=True,
            data_parallel=False,
            comment=None,
            copy_env=True,
            copy_dirs=[],
            max_num_jobs=None,
            num_copies=1,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            dependencies=[],
            repo_name="olmoe-core",
            conda_env_name=SPECS.get("CONDA_ENV_NAME"),
            include_jobs_indices=include_jobs_indices,
            filter_succeeded=filter_succeeded,
            filter_running=filter_running,
            # append_to_sbatch_str=None,
        )
        
    else:
        SWEEP_NAME = sweep_name
        
        if model_type != "DataDecide":
            raise ValueError(f"Model type {model_type} not supported. Only DataDecide is supported at the moment.")

        if models and (model_train_data_sets or model_sizes):
            raise ValueError("Cannot specify both models and model_train_data_sets or model_sizes")

        if model_revisions is not None and (model_seeds or model_steps):
            raise ValueError("Cannot specify both model_revisions and model_seeds or model_steps")
        

        models = [
            DD_MODEL_NAME_TEMPLATE.format(train_data=model_train_data, size=model_size)
            # for model_train_data, model_size in zip(model_train_data_sets, model_sizes)
            for model_train_data, model_size in itertools.product(model_train_data_sets, model_sizes)
        ] if not models else models

        def model_short_name(model, revision="", shorten=True):
            if not shorten:
                return f"{model}".replace('/', '--')
            uploader, model_name = model.split('/')
            model_name = model_name.replace('DataDecide-', 'DD-')
            return f"{model_name}".replace('/', '--')

        if add_time_to_name == 'front':
            time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        # for model, revision in itertools.product(models, model_revisions):
        for model in models:
            # if model_steps in DD_MODEL_STEPS_SETS:
            #     model_steps = DD_MODEL_STEPS_SETS[model_steps]
            model_revisions = [
                DD_REVISION_TEMPLATE.format(step=step, seed=seed) 
                # for step in model_steps for seed in model_seeds 
                for step, seed in itertools.product(model_steps, model_seeds)
            ] if (not model_revisions and model_steps is not None and model_seeds is not None) else model_revisions or ["main"]

            model_sweep_name = f"{SWEEP_NAME}_{command}_{model_short_name(model)}" if add_model_to_name == 'end' else SWEEP_NAME
            SPECS = seq_dict_update([PROJECT_SPECS[os.environ.get('USER')], HARDWARE_SPECS_DICT['all'], HARDWARE_SPECS_DICT.get(model, {}).get(partition)])
            SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
            SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
            SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]
            grid = {
                # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
                # combined with the subgrids
                "main_grid": { 
                    "--model_name_or_path": [model],
                    # "--model_revision": [revision],
                    "--model_revision": model_revisions,
                    "--learning_rate": [5e-7, 5e-6, 5e-5, 5e-4],
                    # "--max_train_samples": [14000, 28000], 
                },
                # allows you to bundle multiple hyperparameters together
                "subgrids": {
                    # "1Mtx1": {"--max_train_samples": [1400], "--num_train_epochs": [1],},
                    "1Mtx10": {"--max_train_samples": [1400], "--num_train_epochs": [10],},
                    "1Mtx100": {"--max_train_samples": [1400], "--num_train_epochs": [100],},
                    # "10Mtx1": {"--max_train_samples": [14000], "--num_train_epochs": [1],},
                    "10Mtx10": {"--max_train_samples": [14000], "--num_train_epochs": [10],},
                    "100M_toks": {"--max_train_samples": [140000], "--num_train_epochs": [1],},
                    # "e1x1c1": {"moe_num_experts_list": ["1"]},
                },
            }

            default_grid = seq_dict_update([MODEL_HP_DEFAULTS.get('all'), MODEL_HP_DEFAULTS.get(model, {}),
                COMMAND_HP_DEFAULTS.get("all"), COMMAND_HP_DEFAULTS.get(command, {})
            ])
            
            run_grid(
                grid,
                default_grid=default_grid,
                sweep_name=model_sweep_name,
                specs=SPECS,
                name_keys=SPECS.get("NAME_KEYS", []),
                prefix=command_prefix,
                cmd_launcher=SPECS.get("CMD_LAUNCHER", "accelerate launch"),
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=(job_time if job_time else SPECS.get("JOBTIME", '24:00:00')),      
                include_job_id=False,
                hashname=False,
                replace_jobname_slashes=True,
                sweep_wandb_tags=[command], 
                saveroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                logroot=f"{SPECS['DEFAULT_SAVE_PATH']}/{model_sweep_name}",
                mem_gb=SPECS["MEM_GB"],
                requeue=True,
                data_parallel=False,
                comment=None,
                copy_env=True,
                copy_dirs=[],
                max_num_jobs=None,
                num_copies=1,
                job_id_start=1,
                debug_mode=DEBUG_MODE,
                dry_mode=DRY_MODE,
                dependencies=[],
                repo_name=SPECS.get("REPO_NAME"),
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                bash_setup_script=SPECS.get("BASH_SETUP_SCRIPT"),
                include_jobs_indices=include_jobs_indices,
                filter_succeeded=filter_succeeded,
                filter_running=filter_running,
                # append_to_sbatch_str=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--sweep-name', type=str, default="", help="Name of the sweep. If not specified, will use the current date and time.")
    parser.add_argument('-c', '--command', type=str, default=OPEN_INSTRUCT_COMMANDS[0], choices=OPEN_INSTRUCT_COMMANDS, help="Command to run for each job.")
    parser.add_argument('-rp', '--relaunch-path', type=str, default=None, help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('-rn', '--relaunch-name', type=str, default=None, help="Name of sweep, also base of sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('-t', '--job-time', type=str)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--cpus', type=int)
    parser.add_argument('--mem', type=str)
    parser.add_argument('-i', '--include-jobs-indices', type=str, default=None)
    parser.add_argument('-nf', '--no-filter', action='store_true', help="If set, will not filter out jobs that have already been run in the sweep. Useful for debugging.")
    parser.add_argument('--model-type', type=str, default="DataDecide", help="Source of model. Only DataDecide is supported at the moment.")
    parser.add_argument('--model-train-data-sets', type=str, help="Training data for the model.")
    parser.add_argument('--model-sizes', type=str, help="Size of the model.")
    parser.add_argument('--model-revisions', type=str, default=None, help="Revision of the model. If not specified, will use the main revision.")
    parser.add_argument('--model-seeds', type=str, help="Seed for the model. Used to differentiate runs with different seeds.")
    parser.add_argument('--model-steps', type=str, help="Training steps for the model. Used to differentiate between checkpoints / revisions of the model.")

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        models=None,
        model_type=args.model_type,
        model_train_data_sets=args.model_train_data_sets.split(",") if args.model_train_data else None,
        model_sizes=args.model_sizes.split(",") if args.model_size else None,
        model_revisions=args.model_revisions.split(',') if args.model_revision is not None else None,
        model_seeds=args.model_seeds.split(',') if args.model_seed is not None else None,
        model_steps=args.model_steps.split(',') if args.model_step is not None else None,
        command=args.command,
        relaunch_path=args.relaunch_path,
        relaunch_name=args.relaunch_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode,
        account=args.slurm_account, 
        partition=args.slurm_partition,
        job_time=args.job_time,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        include_jobs_indices=([int(i) for i in args.include_jobs_indices.split(",")] if args.include_jobs_indices else None),
        filter_running=not args.no_filter,
        filter_succeeded=True,
    )
