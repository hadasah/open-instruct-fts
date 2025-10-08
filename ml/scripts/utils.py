import collections.abc
import itertools
import json
import os
import pandas as pd
import pathlib
import subprocess
import time

from copy import deepcopy
from constants import *

def seq_dict_update(ds):
    """
    Recursively update a dict with another dict.
    This is a deep update, meaning that if a key in the first dict
    has a dict as its value, and the second dict has a key with
    the same name, the value in the first dict will be updated
    with the value from the second dict.
    Keys in the second dict are the ones iterated over. 
    If the value in the second dict is not a dict, it will
    overwrite the value in the first dict.
    """
    if len(ds) == 0:
        return {}
    if len(ds) == 1:
        return deepcopy(ds[0])
    d = deepcopy(ds[0])
    for u in ds[1:]:
        if not u:
            continue
        if not d:
            d = deepcopy(u)
        else:
            u = deepcopy(u)
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = seq_dict_update([d.get(k, {}), v])
                else:
                    d[k] = v
    return d

def has_file_been_modified_recently(filepath, recent_threshold_seconds=3600):
    """
    Checks if a file has been modified within a specified recent time threshold.

    Args:
        filepath (str): The path to the file.
        recent_threshold_seconds (int): The number of seconds defining "recently".
                                       Defaults to 3600 seconds (1 hour).

    Returns:
        bool: True if the file was modified within the threshold, False otherwise.
    """
    if not os.path.exists(filepath):
        return False  # File does not exist

    last_modified_time = os.path.getmtime(filepath)
    current_time = time.time()

    return (current_time - last_modified_time) < recent_threshold_seconds


def model_short_name(model, revision="", shorten=True):
    if not shorten:
        return f"{model}".replace('/', '--')
    try:
        uploader, model_name = model.split('/')
        model_name, model_size = model_name.rsplit('-', 1)
        model_train_data = model_name.replace("DataDecide-", "")

        if revision:
            if "step" in revision:
                step_str, seed = revision.split("step")[1].split("-seed-")
            else:
                assert revision == "main", f"Unexpected revision format: {revision}"
                step_str = str(DD_MODEL_SIZES_INFO.get(model_size, {}).get("training_steps", ""))
                seed = "default"
        seed = DD_MODEL_SEEDS_DICT.get(seed.replace("-", " "), seed)
        model_train_data = DD_TRAIN_SETS_SHORT_NAMES.get(model_train_data, model_train_data)
        
        return f"DD-{model_train_data}-{model_size}-{step_str}-{seed}"
    except Exception as e:
        return f"{model}".replace('/', '--')


def unroll_args(d, prefix=''):
    """Unrolls a dict of args into a list of strings."""
    args = {}
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            args = seq_dict_update([args, unroll_args(v, f'{prefix}.{k}' if prefix else k)])
        else:
            if prefix:
                args[f"{prefix}.{k}"] = v
            else:
                args[k] = v
    return args

def fetch_model_paths(main_grid, use_local_model=True):
    from open_instruct.utils import download_from_hf

    if not use_local_model:
        return  # do not download model if use_local_model is False
    
    model_path_lookup = {}
    new_models = []
    if 'model' in main_grid:
        models = main_grid.get('model', main_grid.get('--model', []))
    else:
        model_name_or_paths = main_grid.get('model_name_or_path', main_grid.get('--model_name_or_path'))
        model_revisions = main_grid.get('model_revision', main_grid.get('--model_revision', 'main'))
        models = list(itertools.product(model_name_or_paths, model_revisions))
    for model_name_or_path, model_revision in models:
        if os.path.exists(model_name_or_path) and os.path.isfile(os.path.join(model_name_or_path, 'config.json')):
            path_parts = reversed(pathlib.Path(model_name_or_path).parts)
            for p in path_parts:
                if p.startswith("25") or p.startswith("2025"):  # assuming run ids start with year like 2023 or 2025
                    model_path_lookup[(p, model_revision)] = (model_name_or_path, model_revision)
                    new_models.append((p, model_revision))
                    break
        else:
            hf_cache = (
                os.path.join(os.environ.get('HF_HOME'), "hub")
                if os.environ.get('HF_HOME') 
                else os.environ.get('HF_HUB_CACHE') or os.path.expanduser('~/.cache/huggingface')
            )
            if not os.path.exists(os.path.join(hf_cache, f'models--{model_name_or_path.replace("/", "--")}', 'refs', model_revision)):
                # raise RuntimeError("""
                #                 Model not found in Hugging Face cache. Please download the model first by running:
                #                 `huggingface-cli download {model_name_or_path} --revision {model_revision}`
                #                 or set `use_local_model=False`
                #                 """.format(model_name_or_path=model_name_or_path, model_revision=model_revision))
                subprocess.run(f"huggingface-cli download {model_name_or_path} --revision {model_revision}", shell=True, check=True)
            # download_from_hf(model_name_or_path, model_revision) # first download the model
            model_path_lookup[(model_name_or_path, model_revision)] = (
                download_from_hf(model_name_or_path, model_revision), # then get the path
                "main"
            )
            new_models.append((model_name_or_path, model_revision))
    main_grid['model'] = new_models
    return main_grid, model_path_lookup


def grab_perf_matched_models(
        pretrain_data_set,
        pretrain_model_size,
        pretrain_model_seed,
        pretrain_model_step,
        match_pretrain_data_set,
        threshold_metric_diff=0.1,
        matched_models_dir=os.path.join(USER_PROJECT_SPEC["PROJECT_DIR"], "matched_models"),
):
    with open(
        os.path.join(
            matched_models_dir, 
            pretrain_data_set, 
            pretrain_model_size,
            pretrain_model_seed,
            pretrain_model_step,
            f"perf_matches_ppl_{pretrain_model_size}.json"
    ), 'r') as f:
        perf_matches = [
            {
                "model_size": m_size, 
                "model_seed": m_seed, 
                "model_step": m_seed_dict["matched_step"], 
                "matched_score": m_seed_dict["matched_score"], 
                "score_diff": m_seed_dict["score_diff"]
            } 
            for m_size, m_size_dict in json.load(f)[match_pretrain_data_set].items() 
            for m_seed, m_seed_dict in m_size_dict.items()
        ]
        perf_matches = sorted(perf_matches, key=lambda item: item["score_diff"])
        perf_match_str = ""
        i = 0
        while True:
            if i >= len(perf_matches) or perf_matches[i]["score_diff"] > threshold_metric_diff:
                break
            perf_match_str += f'{match_pretrain_data_set}:{perf_matches[i]["model_size"]}:{perf_matches[i]["model_step"]}:{perf_matches[i]["model_seed"]}'
            i += 1
        print(perf_match_str)
