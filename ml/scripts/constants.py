"""
"""
import os

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])

MODEL_HP_DEFAULTS = {
    "all": {
        "--model_revision": ["main"],
        # "--tokenizer_revision": ["main"],
        # "--tokenizer_name_or_path": ["allenai/gpt-neox-olmo-dolma-v1_5"],
        "--add_bos": [""],
    },
}

COMMAND_HP_DEFAULTS = {
    "all": {
        "--gradient_checkpointing": [""],
        "--seed": ["8"],
        "--with_tracking": ["true"],
        "--report_to": ["wandb"],
        "--logging_steps": [100],
        "--wandb_entity": ["ml-moe"],
        "--wandb_project_name": ["ft-scaling"],
        "--push_to_hub": ["false"], # do not push to hub by default
        "--try_launch_beaker_eval_jobs": ["false"],
        "--try_auto_save_to_beaker": ["false"],
        "--use_flash_attn": [""],
        "--use_slow_tokenizer": [""],
    },
    "finetune": { #effective batch size 128 https://github.com/hadasah/open-instruct-fts/blob/main/docs/tulu3.md#finetuning
        "--max_seq_length": [4096],
        "--preprocessing_num_workers": [128],
        "--per_device_train_batch_size": [4],
        "--gradient_accumulation_steps": [16],
        "--learning_rate": [5e-06], # 5e-6 for 8B, 2e-6 for 70B
        "--lr_scheduler_type": ["linear"],
        "--warmup_ratio": [0.03],
        "--weight_decay": [0.0],
        "--num_train_epochs": [2],
        "--reduce_loss": ["sum"],
        "--dataset_mixer_list": ["allenai/tulu-3-sft-mixture 1.0"], # about 660M toks, 934407 seqs, avg length = 707 
        "--checkpointing_steps": ["epoch"],
        "--dataset_mix_dir": ["output/sft_8b"],
    },
    "dpo_tune_cache": { # effective batch size 128 https://github.com/hadasah/open-instruct-fts/blob/main/docs/tulu3.md#preference-tuning
        "--max_seq_length": [2048],
        "--preprocessing_num_workers": [16],
        "--per_device_train_batch_size": [2],
        "--gradient_accumulation_steps": [32],
        "--learning_rate": [5e-06], # 5e-7 for 8B, 2e-7 for 70B
        "--lr_scheduler_type": ["linear"],
        "--warmup_ratio": [0.1],
        "--weight_decay": [0.0],
        "--num_train_epochs": [1],
        "--gradient_checkpointing": [""],
        "--dataset_mixer_list":["allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0"],
        "--use_lora": ["false"],
        "--dpo_loss_type": ["dpo_norm"],
        "--dpo_beta": [5],
        "--checkpointing_steps": ["1000"],
    },
}


DD_MODEL_NAME_TEMPLATE = "allenai/DataDecide-{train_data}-{size}"
DD_REVISION_TEMPLATE = "step{step}-seed-{seed}"

DD_MODEL_SIZES = [
    "4M", #
    "6M",
    "8M",
    "10M", #
    "14M",
    "16M",
    "20M", #
    "60M", #
    "90M",
    "150M", #
    "300M", #
    "530M",
    "750M",
    "1B", #
]

DD_MODEL_STEPS_SETS = {
    "1B": 67500,
    "750M": 62500,
    "530M": 51250,
    "300M": 45000,
    "150M": 37500,
    "90M": 29901,
    "60M": 29042,
    "20M": 14584,
    "16M": 24432,
    "14M": 21953,
    "10M": 15117,
    "8M": 13039,
    "6M": 9182,
    "4M": 5725,
}

# size, batch_size, hidden_dim, lr, model_size, heads, layers, training_steps, tokens_trained
DD_MODEL_SIZES_INFO = {
    "4M": {
        "batch_size": 32,
        "hidden_dim": 64,
        "learning_rate": 1.4e-02,
        "model_size": 3.7e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 5725,
        "tokens_trained": 0.4e9,
    },
    "6M": {
        "batch_size": 32,
        "hidden_dim": 96,
        "learning_rate": 1.2e-02,
        "model_size": 6.0e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 9182,
        "tokens_trained": 0.6e9,
    },
    "8M": {
        "batch_size": 32,
        "hidden_dim": 128,
        "learning_rate": 1.1e-02,
        "model_size": 8.5e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 13039,
        "tokens_trained": 0.9e9,
    },
    "10M": {
        "batch_size": 32,
        "hidden_dim": 144,
        "learning_rate": 1.0e-02,
        "model_size": 9.9e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 15117,
        "tokens_trained": 1.0e9,
    },
    "14M": {
        "batch_size": 32,
        "hidden_dim": 192,
        "learning_rate": 9.2e-03,
        "model_size": 14.4e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 21953,
        "tokens_trained": 1.4e9,
    },
    "16M": {
        "batch_size": 32,
        "hidden_dim": 208,
        "learning_rate": 8.9e-03,
        "model_size": 16.0e6,
        "heads": 8,
        "layers": 8,
        "training_steps": 24432,
        "tokens_trained": 1.6e9,
    },
    "20M": {
        "batch_size": 64,
        "hidden_dim": 192,
        "learning_rate": 8.4e-03,
        "model_size": 19.1e6,
        "heads": 8,
        "layers": 16,
        "training_steps": 14584,
        "tokens_trained": 1.9e9,
    },
    "60M": {
        "batch_size": 96,
        "hidden_dim": 384,
        "learning_rate": 5.8e-03,
        "model_size": 57.1e6,
        "heads": 12,
        "layers": 16,
        "training_steps": 29042,
        "tokens_trained": 5.7e9,
    },
    "90M": {
        "batch_size": 160,
        "hidden_dim": 528,
        "learning_rate": 4.9e-03,
        "model_size": 97.9e6,
        "heads": 12,
        "layers": 16,
        "training_steps": 29901,
        "tokens_trained": 9.8e9,
    },
    "150M": {
        "batch_size": 192,
        "hidden_dim": 768,
        "learning_rate": 4.2e-03,
        "model_size": 151.9e6,
        "heads": 12,
        "layers": 12,
        "training_steps": 38157,
        "tokens_trained": 15.0e9,
    },
    "300M": {
        "batch_size": 320,
        "hidden_dim": 1024,
        "learning_rate": 3.3e-03,
        "model_size": 320.0e6,
        "heads": 16,
        "layers": 16,
        "training_steps": 45787,
        "tokens_trained": 30.0e9,
    },
    "530M": {
        "batch_size": 448,
        "hidden_dim": 1344,
        "learning_rate": 2.8e-03,
        "model_size": 530.1e6,
        "heads": 16,
        "layers": 16,
        "training_steps": 57786,
        "tokens_trained": 53.0e9,
    },
    "750M": {
        "batch_size": 576,
        "hidden_dim": 1536,
        "learning_rate": 2.5e-03,
        "model_size": 681.3e6,
        "heads": 16,
        "layers": 16,
        "training_steps": 63589,
        "tokens_trained": 75.0e9,
    },
    "1B": {
        "batch_size": 704,
        "hidden_dim": 2048,
        "learning_rate": 2.1e-03,
        "model_size": 1176.8e6,
        "heads": 16,
        "layers": 16,
        "training_steps": 69369,
        "tokens_trained": 100.0e9,
    },
}


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

DD_MODEL_SEEDS = ["default", "small aux 2", "small aux 3", "large aux 2", "large aux 3"]
DD_MODEL_SEEDS_DICT = {
    "default": 0,
    "small aux 2": 1,
    "small aux 3": 2,
    "large aux 2": 3,
    "large aux 3": 4,
}

MODEL_PATH_LOOKUP = {

}

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


PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        "WANDB_PROJECT": "fts",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "fts",
        "BASH_SETUP_SCRIPT": "/mmfs1/home/margsli/.fts.sh",
        "REPO_NAME": "open-instruct-fts",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-a40,gpu-l40",
    },
    ## TODO : Update these paths to your local setup
    "": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        "WANDB_PROJECT": "fts",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "",
        "BASH_SETUP_SCRIPT": "",
        "REPO_NAME": "open-instruct-fts",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "",
        "SLURM_PARTITION": "",
    },
}

FT_BATCH_SIZE = 128
HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 2,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
    },
    # "": {s
}
