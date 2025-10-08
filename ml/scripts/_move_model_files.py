import subprocess
import os
import glob

model_sweep_paths = ["/gscratch/zlab/margsli/gitfiles/open-instruct-fts/models/250918-*"]

for maybe_model_sweep_path in model_sweep_paths:
    for model_sweep_path in glob.glob(maybe_model_sweep_path.strip()):
        for root, dirs, files in os.walk(model_sweep_path.strip()):
            if "config.json" in files and "pytorch_model.bin" in files:
                if "epoch" in root or "step" in root:
                    continue
                elif "final" in root:
                    dir = os.path.dirname(root)
                    subprocess.run(f"mkdir -p {dir}/model/", shell=True)
                    subprocess.run(f"mv {root}/ {dir}/model", shell=True)
                    # for fname in [
                    #     "config.json",
                    #     "generation_config.json",
                    #     "special_tokens_map.json",
                    #     "tokenizer_config.json",
                    #     "vocab.json",
                    #     "merges.txt",
                    #     "tokenizer.json",
                    #     "chat_template.jinja",
                    #     "pytorch_model.bin",
                    # ]:
                        # if fname in files:
                        #     subprocess.run(f"mv {root}/{fname} {root}/final/{fname}", shell=True)