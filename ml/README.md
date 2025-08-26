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
python ml/scripts/train_sweep.py -a $account -p $partition -sn $sweep_name --model-size $model_sizes --model-train-data-sets $model_train_data_sets
```
Adding `--dry` outputs a command for training without actually launching a job. 
Adding `--debug` launches only one job out of the full grid.

## Setup Eval

I was terrified installing OLMES might ruin the rest of my conda env, so I decided to follow the OLMES README and make a new conda environment:
```bash
git clone https://github.com/allenai/olmes.git
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
model_path=/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_22-10_29_43_test_finetune_DD-dolma1_7-4M/model/2025_08_22-10_29_43_test_finetune_DD-dolma1_7-90M_1Mtx10_--learning_rate=5e-07
results_output_dir=tmp
python -m oe_eval.run_eval --task '{"task_name": "arc_easy", "split": "test", "primary_metric": "acc_per_char", "num_shots": 5, "limit": 1000000000000, "fewshot_source": "OLMES:ARC-Easy", "metadata": {"description": "ARC-Easy (RC) using OLMES-v0.1", "regimes": ["OLMES-v0.1"], "alias": "arc_easy:rc::olmes"}}' --output-dir $results_output_dir --save-raw-requests true --num-workers 1 --model-path ${model_path}

```

Alternatively, I wrote a script to eval every single model in a subdirectory of a given path, and saves the results in the same folder as the models:
```bash
model_sweep_paths=/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-4M_main,/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-10M_main,/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-60M_main,/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=core_9mcqa:rc::olmes:full
python ml/scripts/launch_olmes_evals.py --task ${task} --model-sweep-paths ${model_sweep_paths} --use-all-ckpts

model_sweep_paths=/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=mmlu::olmes
primary_metric=acc_raw
python ml/scripts/launch_olmes_evals.py --task ${task} --primary-metric acc_raw --model-sweep-paths ${model_sweep_paths} --use-all-ckpts

model_sweep_paths=/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/2025_08_21-15_03_00_test_finetune_DD-dolma1_7-150M_main
task=tulu_3_dev
python ml/scripts/launch_olmes_evals.py --task ${task} --model-sweep-paths ${model_sweep_paths} --use-all-ckpts --model-args 
```
To launch this as a job on slurm:
```bash

```
