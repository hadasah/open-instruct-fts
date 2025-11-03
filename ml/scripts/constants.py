"""
"""
import os
from copy import copy

DEFAULT_DIR_PATH = '/'.join(os.path.normpath(os.path.realpath(__file__)).split(os.path.sep)[:-3])

USER = os.getenv("USER")

PROJECT_SPECS = {
    "margsli": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        "WANDB_PROJECT": "ft-scaling",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "fts",
        "BASH_SETUP_SCRIPT": "/mmfs1/home/margsli/.fts.sh",
        "REPO_NAME": "open-instruct-fts",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-a40,gpu-l40",
        "DATASET_LOCAL_CACHE_DIR": "/gscratch/zlab/margsli/.cache/open-instruct/datasets",
    },
    ## TODO : Update these paths to your local setup
    "": {
        'DEFAULT_SAVE_PATH': os.path.join(DEFAULT_DIR_PATH, 'models'),
        "WANDB_PROJECT": "ft-scaling",
        "WANDB_ENTITY": "ml-moe",
        "CONDA_ENV_NAME": "",
        "BASH_SETUP_SCRIPT": "",
        "REPO_NAME": "open-instruct-fts",
        "PROJECT_DIR": DEFAULT_DIR_PATH,
        "SLURM_ACCOUNT": "",
        "SLURM_PARTITION": "",
        "DATASET_LOCAL_CACHE_DIR": "",
    },
}

USER_PROJECT_SPEC = PROJECT_SPECS.get(USER, {})

MODEL_HP_DEFAULTS = {
    "all": {
        # "model_revision": ["main"],
        # "tokenizer_revision": ["main"],
        # "tokenizer_name_or_path": ["allenai/gpt-neox-olmo-dolma-v1_5"],
        "add_bos": [""],
    },
}

COMMAND_HP_DEFAULTS = {
    "all": {
        "gradient_checkpointing": [""],
        "seed": ["8"],
        "with_tracking": ["true"],
        "report_to": ["wandb"],
        "logging_steps": [100],
        "wandb_entity": ["ml-moe"],
        "wandb_project_name": ["ft-scaling"],
        "push_to_hub": ["false"], # do not push to hub by default
        "try_launch_beaker_eval_jobs": ["false"],
        "try_auto_save_to_beaker": ["false"],
        "use_flash_attn": [""],
        "use_slow_tokenizer": [""],
        "dataset-local-cache-dir": [PROJECT_SPECS[USER]["DATASET_LOCAL_CACHE_DIR"]],
    },
    "finetune": { #effective batch size 128 https://github.com/hadasah/open-instruct-fts/blob/main/docs/tulu3.md#finetuning
        "max_seq_length": [4096],
        "preprocessing_num_workers": [128],
        "per_device_train_batch_size": [4],
        "gradient_accumulation_steps": [16],
        "learning_rate": [5e-06], # 5e-6 for 8B, 2e-6 for 70B
        "lr_scheduler_type": ["linear"],
        "warmup_ratio": [0.03],
        "weight_decay": [0.0],
        "num_train_epochs": [2],
        "reduce_loss": ["sum"],
        "dataset_mixer_list": ["allenai/tulu-3-sft-mixture 1.0"], # about 660M toks, 934407 seqs, avg length = 707 
        # "checkpointing_steps": ["epoch"],
        "checkpointing_steps": ["1000"],
        "dataset_mix_dir": ["output/sft_8b"],
    },
    "dpo_tune_cache": { # effective batch size 128 https://github.com/hadasah/open-instruct-fts/blob/main/docs/tulu3.md#preference-tuning
        "max_seq_length": [2048],
        "preprocessing_num_workers": [16],
        "per_device_train_batch_size": [2],
        "gradient_accumulation_steps": [32],
        "learning_rate": [5e-06], # 5e-7 for 8B, 2e-7 for 70B
        "lr_scheduler_type": ["linear"],
        "warmup_ratio": [0.1],
        "weight_decay": [0.0],
        "num_train_epochs": [1],
        "gradient_checkpointing": [""],
        "dataset_mixer_list":["allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0"], # 269478 seqs, 2106 batches of 128
        "use_lora": ["false"],
        "dpo_loss_type": ["dpo_norm"],
        "dpo_beta": [5],
        "checkpointing_steps": ["500"],
    },
}

DD_MODEL_NAME_TEMPLATE = "allenai/DataDecide-{train_data}-{size}"
DD_REVISION_TEMPLATE = "step{step}-seed-{seed}"
DD_SEQ_LEN = 2048

DD_MODEL_STEPS_SETS = {
    "4M": 5725,
    "6M": 9182,
    "8M": 13039,
    "10M": 15117,
    "14M": 21953,
    "16M": 24432,
    "20M": 14584,
    "60M": 29042,
    "90M": 29901,
    "150M": 37500, #diff
    "300M": 45000, #diff
    "530M": 51250, #diff
    "750M": 62500, #diff
    "1B": 67500, #diff
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
        "matched_models": { 
            "dolma1_7": {"6M": [], "8M": [], "10M": [], "14M": [], "16M": [], "20M": [], "60M": [], },
            # :{"6M": [], "8M": [], "10M": [], "14M": [], "16M": [], "20M": [], "60M": [], },
        },
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
        "matched_models": { 
            "dolma1_7": {"8M": [], "10M": [6_250, 7_500], "14M": [2_500, 3_750], "16M": [2_500, 3_750], "20M": [], "60M": []},
            # :{"8M": [], "10M": [], "14M": [], "16M": [], "20M": [], "60M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"10M": [11,250, 12,500, 13,750], "14M": [8,750, 10,000, 11,250], "16M": [8,750, 10,000, 11,250, 12,500], "20M": [3,750], "60M": [1_250],},
            # :{"10M": [], "14M": [], "16M": [], "20M": [], "60M": [], "90M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"14M": [("default", 11_250), ("small aux 2", 11_250), ("small aux 3", 11_250)], "16M": [("default", 12_500), ("small aux 3", 10_000)], "20M": [("default", 5_000), ("small aux 3", 5_000)], "60M": [("default", 1_250), ("small aux 3", 1_250)]},
            # :{"14M": [], "16M": [], "20M": [], "60M": [], "90M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"16M": [], "20M": [], "60M": [], "90M": []},
            # :{"16M": [], "20M": [], "60M": [], "90M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"20M": [], "60M": [], "90M": [], "150M": []},
            # :{"20M": [], "60M": [], "90M": [], "150M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"60M": [], "90M": [], "150M": [], "300M": []},
            # :{"60M": [], "90M": [], "150M": [], "300M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"90M": [("default", 15_000), ("small aux 2", 15_000), ("small aux 3", 15_000)], "150M": [("small aux 3", 7_500)], "300M": [("default", 2_500), ("small aux 2", 2_500), ("small aux 3", 2_500)]},
            # :{"90M": [], "150M": [], "300M": [], "530M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"150M": [], "300M": [], "530M": [], "750M": []},
            # :{"150M": [], "300M": [], "530M": [], "750M": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"300M": [], "530M": [], "750M": [], "1B": []},
            # :{"300M": [], "530M": [], "750M": [], "1B": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"530M": [], "750M": [], "1B": []},
            # :{"530M": [], "750M": [], "1B": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"750M": [], "1B": []},
            # :{"750M": [], "1B": []},
        },
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
        "matched_models": { 
            "dolma1_7": {"1B": []},
            # :{"1B": []},
        },
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

DD_TRAIN_SETS = {
    "dolma1_7": "Dolma1.7",
    "dolma1_7-no-code": "Dolma1.7 (no code)",
    "dolma1_7-no-math-code": "Dolma1.7 (no math, code)",
    "dolma1_7-no-reddit": "Dolma1.7 (no Reddit)",
    "dolma1_7-no-flan": "Dolma1.7 (no Flan)",
    "dolma1_6plus": "Dolma1.6++",
    "c4": "C4",
    "fineweb-pro": "FineWeb-Pro",
    "fineweb-edu": "FineWeb-Edu",
    "falcon": "Falcon",
    "falcon-and-cc": "Falcon+CC",
    "falcon-and-cc-qc-10p": "Falcon+CC (QC 10%)",
    "falcon-and-cc-qc-20p": "Falcon+CC (QC 20%)",
    "falcon-and-cc-qc-orig-10p": "Falcon+CC (QC Orig 10%)",
    "falcon-and-cc-qc-tulu-10p": "Falcon+CC (QC Tulu 10%)",
    "dclm-baseline": "DCLM-Baseline",
    "dclm-baseline-qc-7p-fw2": "DCLM-Baseline (QC 7%, FW2)",
    "dclm-baseline-qc-7p-fw3": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline-qc-fw-3p": "DCLM-Baseline (QC FW 3%)",
    "dclm-baseline-qc-fw-10p": "DCLM-Baseline (QC FW 10%)",
    "dclm-baseline-qc-10p": "DCLM-Baseline (QC 10%)",
    "dclm-baseline-qc-20p": "DCLM-Baseline (QC 20%)",
    "dclm-baseline-25p-dolma1.7-75p": "DCLM-Baseline 25% / Dolma 75%",
    "dclm-baseline-50p-dolma1.7-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dclm-baseline-75p-dolma1.7-25p": "DCLM-Baseline 75% / Dolma 25%",
}

DD_TRAIN_SETS_SHORT_NAMES = {
    "dolma1_7": "d17",
    "dclm-baseline": "dclm",
    "dclm-baseline-qc-7p-fw2": "dclm_qc7fw2",
    "dclm-baseline-qc-7p-fw3": "dclm_qc7fw3",
    "dclm-baseline-qc-fw-3p": "dclm_qcfw3p",
    "dclm-baseline-qc-fw-10p": "dclm_qcfw10p",
    "dclm-baseline-qc-10p": "dclm_qc10p",
    "dclm-baseline-qc-20p": "dclm_qc20p",
    "dclm-baseline-25p-dolma1.7-75p": "dclm_25_d17_75",
    "dclm-baseline-50p-dolma1.7-50p": "dclm_50_d17_50",
    "dclm-baseline-75p-dolma1.7-25p": "dclm_75_d17_25",
}
DD_TRAIN_SETS_SHORT_NAMES_REV = {v: k for k, v in DD_TRAIN_SETS_SHORT_NAMES.items()}

DD_TRAIN_SET_GROUPS = {
    'dclm-dolma': [
        "dclm-baseline",
        "dclm-baseline-25p-dolma1.7-75p",
        "dclm-baseline-50p-dolma1.7-50p",
        "dclm-baseline-75p-dolma1.7-25p",
        "dolma1_7",
    ],
    'dolma-qc': [
        "dclm-baseline",
        "dclm-baseline-qc-7p-fw2",
        "dclm-baseline-qc-7p-fw3",
        "dclm-baseline-qc-fw-3p",
        "dclm-baseline-qc-fw-10p",
        "dclm-baseline-qc-10p",
        "dclm-baseline-qc-20p",
    ],
}
DD_MODEL_SEEDS = ["default", "small aux 2", "small aux 3", "large aux 2", "large aux 3"]
DD_MODEL_SEEDS_DICT = {
    "default": 0,
    "small aux 2": 1,
    "small aux 3": 2,
    "large aux 2": 3,
    "large aux 3": 4,
}

MODEL_PATH_LOOKUP = {
    #"dd__{train_data}-{size}__{revision}__{id/hps}": "path/to/model/",
    # "dd__allenai/DataDecide-{train_data}-{size}__step{step}-seed-{seed}": 
    "dd__dolma1_7-4M__main__100Mt_lr=5e-06": "2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main/model/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main_100M_toks_--learning_rate=5e-06/",
    "dd__dolma1_7-10M__main__100Mt_lr=5e-06": "2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main/model/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-10M_main_100M_toks_--learning_rate=5e-06/",
    "dd__dolma1_7-60M__main__100Mt_lr=5e-06": "2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main/model/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-60M_main_100M_toks_--learning_rate=5e-06/",
    "dd__dolma1_7-150M__main__100Mt_lr=5e-06": "2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main/model/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main_100M_toks_--learning_rate=5e-06/",
}

HP_SHORT_NAMES = {
    "learning_rate": "lr",
}

OPEN_INSTRUCT_COMMANDS = {
    "finetune": {
        "tokens": 660e6,
        "sequences": 934407,
        "batch_size": 128,
    }, 
    "dpo_tune_cache": {
        "tokens": 210e6,
        "sequences": 269478,
    },
    "grpo_fast": {},
    "ppo_fast": {},
    "grpo_vllm_thread_ray_gtrl": {},
    "ppo2": {},
    "ppo_vllm_thread_ray_gtrl": {},
    "reward_modeling": {},
}

FT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    "mmlu",
    "olmes_10_macro_avg",
]

FT_BATCH_SIZE = 128
HARDWARE_SPECS_DICT = {
    "all": {
        "NUM_GPUS": 2,
        "NUM_CPUS": 5,
        "MEM_GB": 120,
    },
    # "": {s
}

BASE_SUITES = {
    "core_9mcqa": [
        "arc_challenge",
        "arc_easy",
        "boolq",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    ],
    "mmlu": [
        "mmlu_abstract_algebra",
        "mmlu_anatomy",
        "mmlu_astronomy",
        "mmlu_business_ethics",
        "mmlu_clinical_knowledge",
        "mmlu_college_biology",
        "mmlu_college_chemistry",
        "mmlu_college_computer_science",
        "mmlu_college_mathematics",
        "mmlu_college_medicine",
        "mmlu_college_physics",
        "mmlu_computer_security",
        "mmlu_conceptual_physics",
        "mmlu_econometrics",
        "mmlu_electrical_engineering",
        "mmlu_elementary_mathematics",
        "mmlu_formal_logic",
        "mmlu_global_facts",
        "mmlu_high_school_biology",
        "mmlu_high_school_chemistry",
        "mmlu_high_school_computer_science",
        "mmlu_high_school_european_history",
        "mmlu_high_school_geography",
        "mmlu_high_school_government_and_politics",
        "mmlu_high_school_macroeconomics",
        "mmlu_high_school_mathematics",
        "mmlu_high_school_microeconomics",
        "mmlu_high_school_physics",
        "mmlu_high_school_psychology",
        "mmlu_high_school_statistics",
        "mmlu_high_school_us_history",
        "mmlu_high_school_world_history",
        "mmlu_human_aging",
        "mmlu_human_sexuality",
        "mmlu_international_law",
        "mmlu_jurisprudence",
        "mmlu_logical_fallacies",
        "mmlu_machine_learning",
        "mmlu_management",
        "mmlu_marketing",
        "mmlu_medical_genetics",
        "mmlu_miscellaneous",
        "mmlu_moral_disputes",
        "mmlu_moral_scenarios",
        "mmlu_nutrition",
        "mmlu_philosophy",
        "mmlu_prehistory",
        "mmlu_professional_accounting",
        "mmlu_professional_law",
        "mmlu_professional_medicine",
        "mmlu_professional_psychology",
        "mmlu_public_relations",
        "mmlu_security_studies",
        "mmlu_sociology",
        "mmlu_us_foreign_policy",
        "mmlu_virology",
        "mmlu_world_religions",
    ],
    "minerva_math": [
        "minerva_math_algebra", 
        "minerva_math_counting_and_probability",  
        "minerva_math_geometry", 
        "minerva_math_intermediate_algebra", 
        "minerva_math_number_theory", 
        "minerva_math_prealgebra", 
        "minerva_math_precalculus", 
    ],
    "bbh": [
        "bbh_boolean_expressions", 
        "bbh_causal_judgement", 
        "bbh_date_understanding", 
        "bbh_disambiguation_qa",
        "bbh_dyck_languages", 
        "bbh_formal_fallacies", 
        "bbh_geometric_shapes", 
        "bbh_hyperbaton", 
        "bbh_logical_deduction_five_objects",  
        "bbh_logical_deduction_seven_objects", 
        "bbh_logical_deduction_three_objects", 
        "bbh_movie_recommendation", 
        "bbh_multistep_arithmetic_two", 
        "bbh_navigate", 
        "bbh_object_counting",  
        "bbh_penguins_in_a_table",  
        "bbh_reasoning_about_colored_objects", 
        "bbh_ruin_names", 
        "bbh_salient_translation_error_detection", 
        "bbh_snarks", 
        "bbh_sports_understanding", 
        "bbh_temporal_sequences", 
        "bbh_tracking_shuffled_objects_five_objects", 
        "bbh_tracking_shuffled_objects_seven_objects", 
        "bbh_tracking_shuffled_objects_three_objects", 
        "bbh_web_of_lies", 
        "bbh_word_sorting",
    ],

}

TASK_SUITE_NAMES = [
    "core_9mcqa:rc::olmes:full",
    "bbh:cot-v1::tulu",
    "bbh:cot-v1::olmes"
    "mmlu:0shot_cot::tulu3",
    "mmlu:rc::olmes",
    "mmlu:mc::tulu",
    "mmlu:mc::olmes", 
    "mmlu:cot::none", 
    "minerva_math::tulu",
    "minerva_math::llama3", 
    "minerva_math::olmes", 
    "minerva_math::bpb", 
]

TASKS_SUITES = {
    "tulu_3_dev": [
        "gsm8k::tulu",
        "drop::llama3",
        "minerva_math::tulu",
        "codex_humaneval::tulu",
        "codex_humanevalplus::tulu",
        "ifeval::tulu",
        "popqa::tulu",
        "mmlu:mc::tulu",
        "alpaca_eval_v2::tulu",
        "bbh:cot-v1::tulu",
        "truthfulqa::tulu",
    ],
    "core_9mcqa:rc::olmes:full": [
        "arc_challenge:rc::olmes:full",
        "arc_easy:rc::olmes:full",
        "boolq:rc::olmes:full",
        "csqa:rc::olmes:full",
        "hellaswag:rc::olmes:full",
        "openbookqa:rc::olmes:full",
        "piqa:rc::olmes:full",
        "socialiqa:rc::olmes:full",
        "winogrande:rc::olmes:full",
    ],
}
TASKS = copy(TASKS_SUITES)
for tsn in TASK_SUITE_NAMES:
    tsn_pre, tsn_post = tsn.split(":", 1)
    if tsn_pre in BASE_SUITES:
        TASKS[tsn] = [base_task + ":" + tsn_post for base_task in BASE_SUITES[tsn_pre]]