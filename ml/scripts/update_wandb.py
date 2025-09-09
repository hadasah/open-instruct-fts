import argparse
import copy
import glob
import json
import os
import pandas as pd
import pathlib
import wandb

from constants import *

_parser = argparse.ArgumentParser()
_parser.add_argument("--model-sweep-paths", default="", type=str, help="Path to top-level folder with models")
_parser.add_argument("--model-sweep-names", default="", type=str, help="Path to top-level folder with models")
_parser.add_argument("--wandb-override", action="store_true",)
_parser.add_argument("--wandb-entity", type=str, default="ml-moe")
_parser.add_argument("--wandb-project", type=str, default="ft-scaling")
_parser.add_argument(
    "--use-all-ckpts",
    action="store_true",
    help="When using --model-sweep-paths, use all checkpoints, not only checkpoints with 'final' in the path",
)


def main():
    args = _parser.parse_args()
    args_dict = vars(args)

    models = args_dict.get("models", "").split(",") if args_dict.get("models", "") != "" else []
    revisions = args_dict.get("revisions", "").split(",") if args_dict.get("revisions", "") != "" else ["main"] * len(models)
    for i, maybe_model in enumerate(models):
        if ":" in maybe_model:
            [models[i], revisions[i]] = maybe_model.split(":", 1)
    model_paths = args_dict.get("model_paths", "").split(",") if args_dict.get("model_paths", "") != "" else []
    
    model_sweep_names = args_dict.get("model_sweep_names", "").split(",") if args_dict.get("model_sweep_names", "") != "" else []
    model_sweep_paths = args_dict.get("model_sweep_paths", "").split(",")  if args_dict.get("model_sweep_paths", "") != "" else []
    assert not (len(model_sweep_paths) > 0 and len(model_sweep_names) > 0), "Cannot specify both --model-sweep-paths and --model-sweep-names!"
    if len(model_sweep_names) > 0:
        for i, model_sweep_name in enumerate(model_sweep_names):
            model_sweep_paths.append(os.path.join(USER_PROJECT_SPEC["DEFAULT_SAVE_PATH"], model_sweep_name))
    
    for maybe_model_sweep_path in model_sweep_paths:
        for model_sweep_path in glob.glob(maybe_model_sweep_path.strip()):
            for root, dirs, files in os.walk(model_sweep_path.strip()):
                if "config.json" in files and "pytorch_model.bin" in files:
                    # if args.use_all_ckpts or "final" in root:
                    model_paths.append(root)
                elif os.path.isfile(os.path.join(root, "wandb/latest-run/files/output.log")):
                    model_paths.append(root)
        assert not (len(models) > 0 and len(model_paths) > 0), "Cannot specify both --models and --model-paths!"
        print(models, revisions, model_paths)

    ppl_df = pd.read_parquet("hf://datasets/allenai/DataDecide-ppl-results/data/train-00000-of-00001.parquet")
    eval_df = pd.read_parquet("hf://datasets/allenai/DataDecide-eval-results/data/macro_avg-00000-of-00001.parquet")
    num_ft_tasks = len(FT_TASKS)

    for model_path in model_paths:
        model_args_dict = copy.deepcopy(args_dict)
        model_args_dict["model_path"] = model_path.strip()
        if model_args_dict.get("output_dir", None) is None:
            model_args_dict["output_dir"] = os.path.join(
                model_args_dict["model_path"], "eval_results"
            )
        else:
            model_args_dict["output_dir"] = os.path.join(
                model_args_dict["output_dir"],
                model_args_dict["model_path"].replace("/", "_"),
            )

        path_parts = reversed(pathlib.Path(model_path).parts)
        run_id = ""
        for p in path_parts:
            if "25" in p:
                run_id = p
                break

        wandb_run_path = f"{args_dict['wandb_entity']}/{args_dict['wandb_project']}/runs/{run_id}"
        wandb_run = wandb.Api().run(wandb_run_path)
        # if key not in wandb_run.config or args_dict["wandb_override"]:
        
        model_info = {}
        config = {}

        command = "finetune" if "finetune" in run_id else "dpo_tune_cache"
        model_size_name = None
        try:
            run_id_parts = run_id.split("_")
            run_id_part = (
                run_id_parts[-3] + "_" + run_id_parts[-2] 
                if "dclm" in run_id_parts[-3] or "dolma" in run_id_parts[-3]
                else run_id_parts[-2] 
            )
            _, data, model_size_name, step, seed = run_id_part.split("-")
            model_info = DD_MODEL_SIZES_INFO[model_size_name]
            config["model_size"] = int(model_info["model_size"])
            config["pretrain_steps"] = int(step)
            seed = DD_MODEL_SEEDS[int(seed)]
        except:
            for msz in DD_MODEL_SIZES_INFO:
                if f"-{msz}" in run_id:
                    model_info = DD_MODEL_SIZES_INFO[msz]
                    config["model_size"] = int(model_info["model_size"])
                    model_size_name = msz
            if not config.get("model_size"):
                continue
            revision = run_id.split(model_size_name)[1].split("_")[0]
            if "step" in revision:
                step_str, seed = revision.split("step")[1].split("-")
                config["pretrain_steps"] = int(step_str.replace("step", ""))
            else:
                config["pretrain_steps"] = DD_MODEL_SIZES_INFO[model_size_name].get("training_steps")
                seed = "default"
            
        pretrain_data = run_id.split(model_size_name)[0].strip("-").rsplit("DD-")[-1]
        config["pretrain_tokens"] = config["pretrain_steps"] * DD_SEQ_LEN * model_info.get("batch_size", 1)
        config["pretrain_sequences"] = config["pretrain_steps"] * model_info.get("batch_size", 1)
        config["pretrain_compute"] = 6 * config["model_size"]  * config["pretrain_steps"] * DD_SEQ_LEN * model_info.get("batch_size", 1)  # in A100-80GB-GPU days
        print(pretrain_data)
        ppl_df_model_row = ppl_df.query(f"params == '{model_size_name}' and step == {config['pretrain_steps']} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
        if len(ppl_df_model_row) != 1:
            print(f"Expected exactly one row for {model_size_name} at step {config['pretrain_steps']} with seed {seed}, but got {len(ppl_df_model_row)}")
        else:
            ppl_df_model_row = ppl_df_model_row.iloc[0]
            for k in ppl_df_model_row.index:
                if k.startswith("eval/"):
                    config[f"pretrain_{k}"] = ppl_df_model_row[k]
        eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and step == {config['pretrain_steps']} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
        if len(eval_ppl_df_model_rows) != num_ft_tasks:
            print(f"Expected exactly {num_ft_tasks} rows for {model_size_name} at step {config['pretrain_steps']} with seed {seed}, but got {len(eval_ppl_df_model_rows)}")
            eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
            eval_ppl_df_model_rows["diff"] = abs(eval_ppl_df_model_rows["step"] - config["pretrain_steps"])
            new_step = eval_ppl_df_model_rows.sort_values(by=["diff"]).iloc[0].step
            eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and step == {new_step} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
            print(f"Using step {new_step} instead")
        for row in eval_ppl_df_model_rows.itertuples():
            task = row.task
            metrics = json.loads(row.metrics)
            for k, v in metrics.items():
                config[f"pretrain_eval_{task}_{k}"] = v

        if command == 'finetune':
            try:
                config["finetune_unique_sequences"] = wandb_run.config["max_train_samples"] if wandb_run.config.get("max_train_samples") else OPEN_INSTRUCT_COMMANDS.get(command, {}).get("sequences")
                config["finetune_sequences"] = config["finetune_unique_sequences"] * int(wandb_run.config.get("num_train_epochs", 1))
                # config["finetune_compute"] = 6 * config["model_size"] * config["finetune_sequences"] * DD_SEQ_LEN # in A100-80GB-GPU days
                config["total_compute_est"] = config["pretrain_compute"] + 6 * config["model_size"] * config["finetune_sequences"] * DD_SEQ_LEN # in A100-80GB-GPU days
            except:
                import pdb; pdb.set_trace()
        elif command == 'dpo_tune_cache':
            pass
    # import pdb; pdb.set_trace()

        for k, v in config.items():
            if k not in wandb_run.config or args_dict["wandb_override"]:
                wandb_run.config[k] = v
        wandb_run.update()
        print(f"Logged metrics to {wandb_run.url}")

if __name__ == "__main__":
    main()
