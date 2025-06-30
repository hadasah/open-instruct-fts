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

Optionally, make a `~/.fts.sh` file that you can source from:
```bash
export ENV_NAME=fts

module load cuda/12.4.1
# module load gcc/13.2.0
mamba activate $ENV_NAME
```

## Testing, Debugging, and Launching jobs interactively

```bash
$account=
$partition=
srun -p $partition --account $account --time=8:00:00 --ntasks-per-node 2 --gpus=2 --cpus-per-task 5 --pty /bin/bash


```