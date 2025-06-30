import argparse
import os
from datetime import datetime
from copy import copy
from slurm_job import run_grid, PROJECT_SPECS_DICT, SPECS_DICT

SWEEP_NAME_DEFAULT = ''
project = 'moe'
MODELS = ["25m"]

def main(
    sweep_name=SWEEP_NAME_DEFAULT,
    add_time_to_name='front',
    add_model_to_name='end',
    debug=False, 
    dry_mode=False,
    account=None, 
    partition=None,
    restart=False,
    include_jobs=None,
):
    if account is None or partition is None:
        raise RuntimeError("Must specify account and partition")

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    PROJECT_SPECS = PROJECT_SPECS_DICT[project]
    SWEEP_NAME = sweep_name
    if add_time_to_name == 'front':
        time_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        SWEEP_NAME = f"{time_str}_{SWEEP_NAME}"

    for model in MODELS:
        model_sweep_name = f"{SWEEP_NAME}_{model}" if add_model_to_name == 'end' else SWEEP_NAME
        SPECS = copy(PROJECT_SPECS["default"])
        SPECS.update(PROJECT_SPECS[model])
        SPECS.update(SPECS_DICT[model][partition])
        grids = {
            model_sweep_name: {
                'model': {
                    'mlp_ratio': [1],
                    'block_type': ['moe'],
                    'activation_type': ['swiglu'],
                    'moe_top_k': [8],
                    'moe_num_experts': [64],
                    'moe_dropless': ['true'],
                    'moe_mlp_impl': ['sparse'], # ['grouped']
                    'moe_zloss_weight': [0.001],
                    'moe_loss_weight': [0.01],
                },
                'optimizer': {
                    'learning_rate': [4.0e-4],
                },
                'scheduler': {
                    't_warmup': [10485760000],
                    't_max': [5e12],
                },
            },
        }

        for sweep_name, grid in grids.items():
            run_grid(
                grid,
                sweep_name=sweep_name,
                name_keys=SPECS.get("NAME_KEYS", []),
                user=os.environ['USER'],
                prefix=SPECS['COMMAND_PREFIX'],
                gpus=SPECS['NUM_GPUS'],
                cpus=SPECS["NUM_CPUS"],
                nodes=((SPECS['NUM_GPUS'] - 1) // 8 + 1),
                node_exclude=None,
                account=account,
                partition=partition,
                DIR_PATH=SPECS["PROJECT_DIR"],
                jobtime=('1:00:00' if debug else SPECS.get("JOBTIME", '24:00:00')),            
                include_job_id=False,
                hide_keys={},
                hashname=False,
                saveroot=f"{SPECS['MODEL_DIR']}/{SWEEP_NAME}",
                logroot=f"{SPECS['MODEL_DIR']}/{SWEEP_NAME}",
                mem_gb=SPECS["MEM_GB"],
                requeue=True,
                data_parallel=False,
                comment=None,
                volta=False,
                volta32=False,
                copy_env=True,
                copy_dirs=[],
                max_num_jobs=None,
                num_copies=1,
                job_id_start=1,
                debug_mode=DEBUG_MODE,
                dry_mode=DRY_MODE,
                add_name='end',
                dependencies=[],
                repo_name="olmoe",
                wandb_project=SPECS.get("WANDB_PROJECT"),
                wandb_entity=SPECS.get("WANDB_ENTITY"),
                conda_env_name=SPECS.get("CONDA_ENV_NAME"),
                include_jobs=include_jobs,
                restart=restart,
                default_config_file=SPECS.get("DEFAULT_CONFIG_FILE"),
                model_config_file=SPECS.get("MODEL_CONFIG_FILE")
                # append_to_sbatch_str=None,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-name', type=str, default=SWEEP_NAME_DEFAULT)
    parser.add_argument('--add-time-to-name', type=str, default='front', choices=['front', 'none'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('-a', '--slurm-account', type=str)
    parser.add_argument('-p', '--slurm-partition', type=str)
    parser.add_argument('-i', '--include-jobs', type=str, default=None)
    parser.add_argument('--restart', action='store_true')

    args = parser.parse_args()

    main(
        sweep_name=args.sweep_name,
        add_time_to_name=args.add_time_to_name,
        debug=args.debug, 
        dry_mode=args.dry_mode, 
        account=args.slurm_account, 
        partition=args.slurm_partition,
        restart=args.restart,
        include_jobs=(args.include_jobs.split(",") if args.include_jobs else None),
    )
