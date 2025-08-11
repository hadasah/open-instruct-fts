"""
"""
import os
from utils import dict_update

#  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])

MODEL_HP_DEFAULTS = {
    "all": {
        "--model_revision": "main",
        "--tokenizer_revision": "main",
    },
    "llama3": {
        "--model_name_or_path": "meta-llama/Llama-3.1-8B",
        "--tokenizer_name": "meta-llama/Llama-3.1-8B "
    },
}


MODEL_NAME_TEMPLATE = "allenai/DataDecide-{train_data}-{size}"
REVISION_TEMPLATE = "step{step}-seed-{seed}"

DD_MODEL_SIZES = [
    "4M", 
    "6M",
    "8M",
    "10M",
    "14M",
    "16M",
    "20M",
    "60M",
    "90M",
    "150M",
    "300M",
    "530M",
    "750M",
    "1B",
]

DD_TRAIN_SETS = [
    "dolma1_7",
    "dolma1_7-no-code",
    "dolma1_7-no-math-code",
    "dolma1_7-no-reddit",
    "dolma1_7-no-flan",
    "dolma1_6plus",
    "c4",
    "fineweb-pro",
    "fineweb-edu",
    "falcon",
    "falcon-and-cc",
    "falcon-and-cc-qc-10p",
    "falcon-and-cc-qc-20p",
    "falcon-and-cc-qc-orig-10p",
    "falcon-and-cc-qc-tulu-10p",
    "dclm-baseline",
    "dclm-baseline-qc-7p-fw2",
    "dclm-baseline-qc-7p-fw3",
    "dclm-baseline-qc-fw-3p",
    "dclm-baseline-qc-fw-10p",
    "dclm-baseline-qc-10p",
    "dclm-baseline-qc-20p",
    "dclm-baseline-25p-dolma1.7-75p",
    "dclm-baseline-50p-dolma1.7-50p",
    "dclm-baseline-75p-dolma1.7-25p",
]

DD_MODEL_SEEDS = ["default", "small-aux-2", "small-aux-3"]

OPEN_INSTRUCT_COMMANDS = [
    "finetune",
    "dpo_tune_cache",
    "grpo_fast",
    "ppo_fast",
    "grpo_vllm_thread_ray_gtrl",
    "ppo2",
    "ppo_vllm_thread_ray_gtrl",
    "reward_modeling",
]

# MODEL_HP_DEFAULTS = dict_update(MODEL_HP_DEFAULTS, {f"allenai/DataDecide-dolma1_7-300M": )

CMD_DEFAULTS = {
    "all": {
        "--use_slow_tokenizer": [""],
        "--use_flash_attn": [""],
        "--gradient_checkpointing": [""],
        "--report_to": ["wandb"],
        "--with_tracking": [""],
        "--logging_steps": ["1"],
        "--seed": ["8"],
        "--push_to_hub": ["false"],
    },
    "ft": {
        "--dataset_mixer_list": ["allenai/tulu-3-sft-mixture 1.0"],
        "--max_seq_length": ["4096"],
        "--per_device_train_batch_size": ["1"],
        "--gradient_accumulation_steps": ["2"],
        "--learning_rate": ["5e-06"],
        "--lr_scheduler_type": ["linear"],
        "--warmup_ratio": ["0.03"],
        "--weight_decay": ["0.0"],
        "--num_train_epochs": ["2"],
        "--reduce_loss": ["sum"],
    },
    "dpo": {
        "--dataset_mixer_list": ["allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0"],
        "--max_seq_length": ["2048"],
        "--gradient_accumulation_steps": ["16"],
        "--learning_rate": ["5e-07"],
        "--lr_scheduler_type": ["linear"],
        "--warmup_ratio": ["0.1"],
        "--weight_decay": ["0.0"],
        "--num_train_epochs": ["1"],
        "--dpo_loss_type": ["dpo_norm"],
        "--dpo_beta": ["5"],
    }
}



PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        "WANDB_PROJECT": "fts",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "fts",
        "REPO_NAME": "open-instruct-fts",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-a40,gpu-l40",
        "MODEL": [],
        "NAME_KEYS": [],
    },
    ## TODO : Update these paths to your local setup
}

BATCH_SIZE = 512
HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 4,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
        "per_gpu_batch_size": 16,
    },
    "olmo2_10M": { 
        "gpu-rtx6k": {
            "per_gpu_batch_size": 16,
        }, 
    },
}
