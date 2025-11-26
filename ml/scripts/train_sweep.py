import argparse
import itertools
import json
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid
from constants import *
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
    relaunch_paths=None,
    relaunch_names=None,
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

    if relaunch_paths or relaunch_names:
        if relaunch_names and relaunch_paths:
            raise ValueError("Cannot specify both relaunch_name and relaunch_path")
        if relaunch_names:
            relaunch_paths = [os.path.join(PROJECT_SPECS[user]['DEFAULT_SAVE_PATH'], relaunch_name) for relaunch_name in relaunch_names.split(",")]
        else:
            relaunch_paths = relaunch_paths.split(",")
        for relaunch_path in relaunch_paths:
            relaunch_path = relaunch_path.rstrip('/')
            model_sweep_name = os.path.basename(relaunch_path)
            path_to_grid_file = os.path.join(relaunch_path, 'grid.json')
            path_to_specs = os.path.join(relaunch_path, 'specs.json')
            if not os.path.exists(path_to_grid_file):
                raise FileNotFoundError(f"Grid file {path_to_grid_file} does not exist.")
            grid = json.load(open(path_to_grid_file, 'r'))
            # model = grid.get('main_grid', {}).get('model_name', [None])[0]
            if 'model' in grid.get('main_grid', {}):
                grid["main_grid"]["model"] = [(mn, mr) for [mn, mr] in grid['main_grid']['model']]
            for subgrid in grid.get('subgrids', {}):
                if 'model' in grid['subgrids'][subgrid]:
                    grid['subgrids'][subgrid]['model'] = [(mn, mr) for [mn, mr] in subgrid['model']]
            # SPECS = seq_dict_update([USER_SPECS, HARDWARE_SPECS_DICT['all'], HARDWARE_SPECS_DICT.get(model, {}).get(partition)])
            SPECS = seq_dict_update([USER_SPECS, HARDWARE_SPECS_DICT['all']])
            SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
            SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
            SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

            if os.path.exists(path_to_specs):
                old_specs = json.load(open(path_to_specs, 'r'))
                for key in old_specs:
                    if key not in ignore_specs_check_keys:
                        assert SPECS.get(key) == old_specs[key], f"Specs mismatch for {key}: {SPECS.get(key)} != {old_specs[key]}"
            
            default_grid = seq_dict_update([
                # MODEL_HP_DEFAULTS.get('all'), MODEL_HP_DEFAULTS.get(model, {}), COMMAND_HP_DEFAULTS.get("all"), COMMAND_HP_DEFAULTS.get(command, {})
                MODEL_HP_DEFAULTS.get('all'), COMMAND_HP_DEFAULTS.get("all"), COMMAND_HP_DEFAULTS.get(command, {})
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
                sweep_name_position="start",
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
        if model_type != "DataDecide":
            raise ValueError(f"Model type {model_type} not supported. Only DataDecide is supported at the moment.")
        if models and (model_train_data_sets or model_sizes or model_seeds or model_steps):
            raise ValueError("If models is specified, cannot specify model_train_data_sets, model_sizes, model_seeds, or model_steps.")
        
        SWEEP_NAME = sweep_name
        models = models.split("::") if models else [f"{model_train_data_sets}:{model_sizes}:{model_steps}:{model_seeds}"]
        sweep_wandb_config = {}

        if add_time_to_name == 'front':
            # time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            time_str = str(datetime.now().strftime('%y%m%d-%H%M%S'))
            SWEEP_NAME = f"{time_str}_{SWEEP_NAME}" if SWEEP_NAME else time_str
        
        model_args_list = []
        for model_str in models:
            model_train_data_set, model_sizes, model_steps, model_seeds = model_str.split(":")
            model_train_data_sets = DD_TRAIN_SET_GROUPS[model_train_data_set] if model_train_data_set in DD_TRAIN_SET_GROUPS else [model_train_data_set]
            model_sizes = model_sizes.split(",") if model_sizes else list(DD_MODEL_SIZES_INFO.keys())
            model_seeds = model_seeds.split(",") if model_seeds else list(DD_MODEL_SEEDS_DICT.keys())
            model_seeds = [DD_MODEL_SEEDS[int(s)] if s.isdigit() and int(s) < len(DD_MODEL_SEEDS) else s for s in model_seeds]
            model_steps = [int(s) for s in model_steps.split(",")] if model_steps else []
            model_names = [
                DD_MODEL_NAME_TEMPLATE.format(train_data=model_train_data, size=model_size)
                # for model_train_data, model_size in zip(model_train_data_sets, model_sizes)
                for model_train_data, model_size in itertools.product(model_train_data_sets, model_sizes)
            ]
            formatted_model_revisions = [
                DD_REVISION_TEMPLATE.format(step=step, seed=seed).replace(" ", "-")
                # for step in model_steps for seed in model_seeds 
                for step, seed in itertools.product(model_steps, model_seeds)
            ] if (not model_revisions and model_steps and model_seeds) else copy.copy(model_revisions) or ["main"]
            print(f"Using model names: {model_names}")
            print(f"Using model revisions: {formatted_model_revisions}")
            model_args_list += [(model_name, model_revision) for model_name, model_revision in itertools.product(model_names, formatted_model_revisions)]

        model_sweep_name = f"{SWEEP_NAME}_{command}"
        SPECS = seq_dict_update([PROJECT_SPECS[os.environ.get('USER')], HARDWARE_SPECS_DICT['all'], HARDWARE_SPECS_DICT.get(model_sizes[0], {}).get(partition)])
        SPECS['NUM_GPUS'] = gpus or SPECS['NUM_GPUS']
        SPECS["NUM_CPUS"] = cpus or SPECS["NUM_CPUS"]
        SPECS["MEM_GB"] = mem or SPECS["MEM_GB"]

        grid = {
            # main_grid is the top-level grid, the sweep will run over all combinations of these hyperparameters, 
            # combined with the subgrids
            "main_grid": { 
                "model": model_args_list,
                "learning_rate": [5e-6], #[5e-5, 5e-6],
                # "max_train_samples": [14000, 28000], 
            },
            # allows you to bundle multiple hyperparameters together
            "subgrids": {
                # "1Mtx1": {"max_train_samples": [1400], "num_train_epochs": [1],},
                # "1Mtx10": {"max_train_samples": [1400], "num_train_epochs": [10],},
                # "1Mtx100": {"max_train_samples": [1400], "num_train_epochs": [100],},
                # "10Mtx1": {"max_train_samples": [14000], "num_train_epochs": [1],},
                # "10Mtx10": {"max_train_samples": [14000], "num_train_epochs": [10],},
                # "100Mtx1": {"max_train_samples": [140000], "num_train_epochs": [1],},
                "Ft": {},
                "2.5Ft": {"num_train_epochs": [5],}
            },
        }
        default_grid = seq_dict_update([MODEL_HP_DEFAULTS.get('all'), #MODEL_HP_DEFAULTS.get(model, {}),
            COMMAND_HP_DEFAULTS.get("all"), COMMAND_HP_DEFAULTS.get(command, {})
        ])
        print(grid)
        
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
            jobtime=(job_time if job_time else SPECS.get("JOBTIME", '72:00:00')),      
            include_job_id=False,
            hashname=False,
            replace_jobname_slashes=True,
            sweep_name_position="start",
            sweep_wandb_tags=[command], 
            sweep_wandb_config=sweep_wandb_config,
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
    parser.add_argument('-c', '--command', type=str, default=list(OPEN_INSTRUCT_COMMANDS.keys())[0], choices=OPEN_INSTRUCT_COMMANDS, help="Command to run for each job.")
    parser.add_argument('-rp', '--relaunch-paths', type=str, default=None, help="Path to the sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
    parser.add_argument('-rn', '--relaunch-names', type=str, default=None, help="Name of sweep, also base of sweep directory containing grid.json and specs.json. Used to restart jobs from a previous sweep.")
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
    parser.add_argument('--models', type=str, default='', help="Semicolon-separated list of models: <train_data>:<sizes>:<steps>:<seeds>;")
    parser.add_argument('--model-train-data-sets', type=str, default='', help="Comma-separated list of training data sets for the model. Used to construct model names.")
    parser.add_argument('--model-sizes', type=str, default='', help="Size of the model.")
    parser.add_argument('--model-revisions', type=str, default='', help="Revision of the model. If not specified, will use the main revision.")
    parser.add_argument('--model-seeds', type=str, default='', help="Seed for the model. Used to differentiate runs with different seeds.")
    parser.add_argument('--model-steps', type=str, default='', help="Training steps for the model. Used to differentiate between checkpoints / revisions of the model.")

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        models=args.models,
        model_type=args.model_type,
        model_train_data_sets=args.model_train_data_sets,
        model_sizes=args.model_sizes,
        model_revisions=args.model_revisions,
        model_seeds=args.model_seeds,
        model_steps=args.model_steps,
        command=args.command,
        relaunch_paths=args.relaunch_paths,
        relaunch_names=args.relaunch_names,
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
        filter_succeeded=not args.no_filter,
    )
