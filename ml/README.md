## Installation

On the UW cluster, load the right modules:

```bash
module load cuda/12.4.1
```

Optionally, create a conda environment:
```bash
export ENV_NAME=fts
mamba create -n $ENV_NAME python=3.11
```

Clone the repo:
```bash
export FTS_DIR=
export MODEL_DIR=${FTS_DIR}/models
git clone https://github.com/hadasah/open-instruct-fts.git $FTS_DIR
cd $FTS_DIR
```

Follow the open-instruct installation instructions:
```bash
pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install packaging
pip install flash-attn==2.7.2.post1 --no-build-isolation
pip install -r requirements.txt
pip install -e .
python -m nltk.downloader punkt
```

Install `ai2-olmo` to allow loading DataDecide models:
```bash
pip install ai2-olmo
```

Fix the `antlr4` version to work with OmegaConf:
```bash
pip install antlr4-python3-runtime==4.9
```

Log in to huggingface cli to access models and data:
```bash
huggingface-cli login 
```
Follow prompts to create/use a token. 
Read-only will work, as long as you don't need to uplaod model checkpoints to huggingface.

Optionally, make a `~/.fts.sh` file that you can source from:
```bash
export ENV_NAME=fts
export FTS_DIR=
export MODEL_DIR=${FTS_DIR}/models
module load cuda/12.4.1
mamba activate $ENV_NAME
```

## Testing, Debugging, and Launching jobs interactively

```bash
account=
partition=
srun -p $partition --account $account --time=8:00:00 --ntasks-per-node 2 --gpus=2 --cpus-per-task 5 --pty /bin/bash
```

## Luanching sweeps / job grids

```bash
account=
partition=
sweep_name=
model_train_data_sets=dolma1_7
model_sizes=4M
python ${FTS_DIR}/ml/scripts/train_sweep.py -a $account -p $partition -sn $sweep_name --model-size $model_sizes --model-train-data-sets $model_train_data_sets
```
Adding `--dry` outputs a command for training without actually launching a job. 
Adding `--debug` launches only one job out of the full grid.

## Setup Eval

I was terrified installing OLMES might ruin the rest of my conda env, so I decided to follow the OLMES README and make a new conda environment:
```bash
git clone https://github.com/hadasah/olmes.git
cd olmes

conda create -n olmes python=3.10
conda activate olmes
pip install -e .
```

## Eval Commands

To use raw OLMES commands directly:
```bash
model_train_data_set=dolma1_7
model_size=150M
model_step=37500
model_seed=default
task=arc_easy:rc::olmes:full
results_output_dir=tmp
olmes --model allenai/DataDecide-${model_train_data_set}-${model_size} --revision step${model_step}-seed-${model_seed} --task ${task} --output-dir ${results_output_dir}


model_train_data_set=dolma1_7
model_size=150M
model_step=37500
model_seed=default
task=tulu_3_dev
results_output_dir=tmp
olmes --model allenai/DataDecide-${model_train_data_set}-${model_size} --revision step${model_step}-seed-${model_seed} --task ${task} --output-dir ${results_output_dir} --model-args '{"chat_template": "tulu"}'

```

The `olmes` command, which runs `launch.py`, does not take model paths. You have to use `run_eval.py` instead: 
```bash
model_path=${MODEL_DIR}/models/2025_08_22-10_29_43_test_finetune_DD-dolma1_7-4M/model/2025_08_22-10_29_43_test_finetune_DD-dolma1_7-90M_1Mtx10_--learning_rate=5e-07
results_output_dir=tmp
python -m oe_eval.run_eval --task '{"task_name": "arc_easy", "split": "test", "primary_metric": "acc_per_char", "num_shots": 5, "limit": 1000000000000, "fewshot_source": "OLMES:ARC-Easy", "metadata": {"description": "ARC-Easy (RC) using OLMES-v0.1", "regimes": ["OLMES-v0.1"], "alias": "arc_easy:rc::olmes"}}' --output-dir $results_output_dir --save-raw-requests true --num-workers 1 --model-path ${model_path}

```

Alternatively, I wrote a script to eval every single model in a subdirectory of a given path, and saves the results in the same folder as the models:
```bash
model_sweep_paths=${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main,${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-10M_main,${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-60M_main,${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=core_9mcqa:rc::olmes:full
python ml/scripts/launch_olmes_evals.py --task ${task} --model-sweep-paths ${model_sweep_paths} --use-all-ckpts

model_sweep_paths=${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=mmlu::olmes
primary_metric=acc_raw
python ml/scripts/launch_olmes_evals.py --task ${task} --primary-metric acc_raw --model-sweep-paths ${model_sweep_paths} --use-all-ckpts

model_sweep_paths=${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=tulu_3_dev

python ${FTS_DIR}/ml/scripts/launch_olmes_evals.py --task tulu_3_dev --models allenai/DataDecide-dolma1_7-4M:step5725-seed-default,allenai/DataDecide-dolma1_7-10M:step15117-seed-default,allenai/DataDecide-dolma1_7-60M:step29042-seed-default,allenai/DataDecide-dolma1_7-150M:step37500-seed-default --use-all-ckpts --model-args chat_template=tulu --output-dir ${MODEL_DIR}/_eval_results

python ${FTS_DIR}/ml/scripts/launch_olmes_evals.py --task tulu_3_dev --model-paths ${MODEL_DIR}/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main/models/ --use-all-ckpts --model-args chat_template=tulu --output-dir ${MODEL_DIR}/_eval_results
```
To launch this as a job on slurm:
```bash
sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task core_9mcqa:rc::olmes:full --model-sweep-paths /gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_30-17_19_37_test_finetune_DD-dolma1_7-150M --log-to-wandb"

sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task core_9mcqa:rc::olmes:full --model-sweep-paths /gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-08_24_43_test_finetune_DD-dolma1_7-4M_main --use-all-ckpts --log-to-wandb"

MODELS=allenai/DataDecide-dolma1_7-4M:step5725-seed-default
MODELS=allenai/DataDecide-dolma1_7-10M:step15117-seed-default
MODELS=allenai/DataDecide-dolma1_7-60M:step29042-seed-default
MODELS=allenai/DataDecide-dolma1_7-150M:step37500-seed-default
sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task tulu_3_dev --models allenai/DataDecide-dolma1_7-150M:step37500-seed-default --use-all-ckpts --model-args chat_template=tulu --output-dir /gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_results"

sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task core_9mcqa:rc::olmes:full --log-to-wandb --model-sweep-names 250909*"

sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task gsm8k::tulu drop::llama3 ifeval::tulu --model-paths /gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/250909-220627_test_match_60M_finetune/250909-220627_test_match_60M_finetune_100Mtx1_DD-dclm_25_d17_75-150M-10000-0_lr=5e-06/model/final"

sbatch shell_scripts/run_script_with_defaults.sh olmes "python /gscratch/zlab/margsli/gitfiles/open-instruct-fts/ml/scripts/launch_olmes_evals.py --task minerva_math::tulu ifeval::tulu codex_humanevalplus --limit 0.1 --model-sweep-names 250909-220627* --filter-string Ft"


gsm8k::olmes minerva_math_algebra::olmes minerva_math_algebra::llama3 codex_humaneval:3shot:bpb::none codex_humaneval:3shot::none codex_humanevalplus::none popqa truthfulqa::olmo1 alpaca_eval_v2 bbh_boolean_expressions:cot-v1::olmes

```

"bbh:cot-v1::tulu", "minerva_math::tulu", "mmlu:mc::tulu",  "gsm8k", "drop", "codex_humanevalplus", "ifeval", "popqa"

#

 core_9mcqa:rc::olmes:full tulu_3_dev gsm8k::olmes drop::llama3 minerva_math_algebra::olmes minerva_math_algebra::llama3 ifeval::tulu codex_humanevalplus codex_humaneval:3shot:bpb::none codex_humaneval:3shot::none codex_humanevalplus::none popqa truthfulqa::olmo1 alpaca_eval_v2 bbh_boolean_expressions:cot-v1::tulu

