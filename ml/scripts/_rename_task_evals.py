import json
import subprocess
import os
import glob


folders = {
    "": "/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_results",
    # "limit0.1_":"/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/_eval_resultslimit0.1", 
    # "1B": "/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/251001-204948_finetune"
}

for folder_key, folder_path in folders.items():
    for maybe_folder in glob.glob(folder_path.strip()):
        for root, dirs, files in os.walk(maybe_folder.strip()):
            if os.path.isdir(root):
                for file in files:
                    if file.endswith("metrics.json"):
                        with open(os.path.join(root, file), "r") as f:
                            line_data = json.loads(f.read())
                            old_task_name = line_data['task_name']
                            new_task_name = line_data.get('task_config', {}).get('metadata', {}).get('alias', line_data['task_name'])
                        for old_fn in glob.glob(os.path.join(root, f"{file[:8]}-{old_task_name}-*")):
                            new_fn = old_fn.replace(old_task_name, new_task_name)
                            os.rename(old_fn, new_fn)