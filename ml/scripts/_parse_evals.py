import subprocess
import os
import json
import pandas as pd
import glob

from constants import TASKS, TASKS_SUITES

def get_all_tasks(suites):
    all_tasks = []
    for suite in suites:
        if suite not in all_tasks:
            all_tasks.append(suite)
        tasks = TASKS_SUITES.get(suite, [])
        # tasks = TASKS.get(suite, [])
        for task in tasks:
            if task not in all_tasks:
                all_tasks.append(task)
    if len(all_tasks) == len(set(suites)):
        return all_tasks
    return get_all_tasks(all_tasks)

task_suites = [
    "core_9mcqa:rc::olmes:full",
    "tulu_3_dev",
    "mmlu:0shot_cot::tulu3",
    "mmlu:rc::olmes",
    "gsm8k::olmes",
    "drop::llama3",
    "minerva_math_algebra::olmes",
    "minerva_math_algebra::llama3",
    "codex_humanevalplus",
    "codex_humaneval:3shot::none",
    "codex_humaneval:3shot::olmo3",
    "codex_humaneval:0-shot-chat",
    "codex_humaneval:0-shot-chat-pass1-sample",
    "codex_humaneval:3shot:bpb::none",
    "codex_humanevalplus::none",
    "popqa",
    "truthfulqa::olmo1",
    "alpaca_eval_v2",
    "bbh_boolean_expressions:cot-v1::olmes",
    "mmlu:mc::olmes", "mmlu:cot::none", "minerva_math::llama3", "minerva_math::olmes", "minerva_math::bpb", "bbh:cot-v1::olmes", "ifeval:0-shot-cot", "popqa", "drop::olmes", "drop:0shot-chat::olmes", "drop:rc::gen2mc", "drop:mc::gen2mc",
    "arc_challenge:mc::olmes", "arc_easy:mc::olmes", "boolq:mc::olmes", "hellaswag:mc::olmes", "openbookqa:mc::olmes", "piqa:mc::olmes", "socialiqa:mc::olmes", "winogrande:mc::olmes"
]
col_order = ["model"] + get_all_tasks(task_suites)


folders = {
    "": "/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_results",
    "cft": "/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_results_cftulu",
    "limit0.1_":"/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_resultslimit0.1", 
    "1B": "/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/251001-204948_finetune"
}

model_names_to_use = []
model_names_to_use = [
    # "meta-llama_Llama-3.1-8B__main",
    # "allenai_Llama-3.1-Tulu-3-8B-SFT__main",
    # "allenai_Llama-3.1-Tulu-3-8B-DPO__main",
    # "meta-llama_Llama-3.1-8B-Instruct__main",
]

df = {}
keys = set()
for folder_key, folder_path in folders.items():
    for maybe_folder in glob.glob(folder_path.strip()):
        for root, dirs, files in os.walk(maybe_folder.strip()):
            if os.path.isdir(root) and "metrics-all.jsonl" in files:
                folder = root
                # prefix = folder.replace("/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_results", "")
                # prefix = prefix + "_" if prefix else ""
                model_name = os.path.basename(folder)
                if 'eval_results' in model_name:
                    model_name = os.path.basename(os.path.dirname(folder))
                if model_name == 'final':
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(folder)))
                if model_name == 'model':
                    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(folder))))
                
                if model_names_to_use and model_name not in model_names_to_use:
                    continue

                tasks = set()
                key = "" # folder_key
                with open(os.path.join(folder, "metrics-all.jsonl"), "r") as f:
                    for line in f:
                        line_data = json.loads(line)
                        task_name = line_data.get('task_config', {}).get('metadata', {}).get('alias', line_data['task_name'])
                        lkey = f"_limit{line_data['task_config']['limit']}" if line_data['task_config'].get('limit', None) else ""
                        keys.add(lkey)
                        suffix = f"_cf{line_data['model_config']['chat_template']}" if line_data.get("model_config", {}).get("chat_template") else ""
                        if f"{task_name}{lkey}{suffix}" in tasks:
                            continue
                        model_name_key = f'{model_name}{suffix}'
                        if model_name_key not in df:
                            df[model_name_key]= {"model": model_name_key, "metrics":[]}
                        df[model_name_key][f"{task_name}{lkey}"] = line_data.get("metrics", {}).get("primary_score", None)
                        tasks.add(f"{task_name}{lkey}{suffix}")
                for model_sweep_path in glob.glob(f"{folder}/task-*metrics.json"):
                    with open(model_sweep_path, "r") as f:
                        for line in f:
                            line_data = json.loads(line)
                            task_name = line_data.get('task_config', {}).get('metadata', {}).get('alias', line_data['task_name'])
                            lkey = f"_limit{line_data['task_config']['limit']}" if line_data['task_config'].get('limit', None) else ""
                            keys.add(lkey)
                            suffix = f"_cf{line_data['model_config']['chat_template']}" if line_data.get("model_config", {}).get("chat_template") else ""
                            if f"{task_name}{lkey}{suffix}" in tasks:
                                continue
                            model_name_key = f'{model_name}{suffix}'
                            if model_name_key not in df:
                                df[model_name_key]= {"model": model_name_key, "metrics":[]}
                            df[model_name_key][f"{task_name}{lkey}"] = line_data.get("metrics", {}).get("primary_score", None)
                            tasks.add(f"{task_name}{lkey}{suffix}")
dfs = []
for k in df:
    dfs.append(df[k])
df = pd.DataFrame(dfs)
print(df.columns)
print(df)
# df.reindex(["Z", "C", "A"])
df = df.set_index('model')
model_order = [
    "meta-llama_Llama-3.1-8B__main",
    "meta-llama_Llama-3.1-8B__main_cftulu",
    "allenai_Llama-3.1-Tulu-3-8B-SFT__main",
    "allenai_Llama-3.1-Tulu-3-8B-SFT__main_cftulu",
    "allenai_Llama-3.1-Tulu-3-8B-DPO__main",
    "allenai_Llama-3.1-Tulu-3-8B-DPO__main_cftulu",
    "meta-llama_Llama-3.1-8B-Instruct__main",
    "meta-llama_Llama-3.1-8B-Instruct__main_cftulu",
]
model_order += [m for m in df.index.tolist() if m not in model_order]
df = df.reindex(model_order, axis=0)
df = df.reset_index()
df = df[[f"{c}{key}" for key in keys for c in col_order if f"{c}{key}" in df.columns]]
cols = sorted(df.columns.tolist()[1:])
print(cols)
df = df[["model"] + cols]
print(df)
df.to_csv(os.path.join(folders[""], "metrics-all.csv"), index=False)