import argparse
import copy
import glob
import json
import numpy as np
import os
import pandas as pd
import pathlib
import wandb

from constants import *
from utils import *

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

def log_sum_exp(log_probs):
    """Numerical stable way to compute log(sum(exp(log_probs)))"""
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))


def check_finite_and_nan(value, name):
    assert np.isfinite(value), f"{name}: {value} is inf or -inf"
    assert not np.isnan(value), f"{name}: {value} is NaN"


def process_predictions_cheap_decisions(prediction):
    GENERATE_PRIMARY_METRICS = [
        "em",
        "f1",
        "exact_match",
        "pass_at_1",
    ]
    metrics = prediction["metrics"]
    model_outputs = prediction["model_output"]

    # 1. RC tasks
    if all(key in metrics for key in ["acc_raw", "acc_per_char"]):
        correct_idx = metrics["correct_choice"]
        correct_output = model_outputs[correct_idx]

        # Compute correct seq
        correct_logit = correct_output["sum_logits"]
        correct_logit_per_token = correct_output["logits_per_token"]
        correct_logit_per_char = correct_output["logits_per_char"]
        # correct_logit_per_byte = correct_output["logits_per_byte"]

        # Compute margin
        correct_prob = np.exp(correct_logit)
        correct_prob_per_token = np.exp(correct_logit_per_token)
        correct_prob_per_char = np.exp(correct_logit_per_char)
        # correct_prob_per_byte = np.exp(correct_logit_per_byte)
        incorrect_probs = [
            np.exp(out["sum_logits"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_token = [
            np.exp(out["logits_per_token"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        incorrect_probs_per_char = [
            np.exp(out["logits_per_char"])
            for i, out in enumerate(model_outputs)
            if i != correct_idx
        ]
        # incorrect_probs_per_byte = [
        #     np.exp(out["logits_per_byte"])
        #     for i, out in enumerate(model_outputs)
        #     if i != correct_idx
        # ]

        # Compute uncond
        if all("sum_logits_uncond" in option for option in model_outputs):
            uncond_logits = np.array(
                [option["sum_logits_uncond"] for option in model_outputs]
            )
            uncond_correct_logit = uncond_logits[correct_idx]
            uncond_correct_prob = np.exp(uncond_correct_logit)
            uncond_correct_prob_per_token = np.exp(
                uncond_correct_logit / correct_output["num_tokens"]
            )
            uncond_correct_prob_per_char = np.exp(
                uncond_correct_logit / correct_output["num_chars"]
            )
            # uncond_correct_prob_per_byte = np.exp(
            #     uncond_correct_logit / correct_output["num_bytes"]
            # )
            # sum
            uncond_total_logit = log_sum_exp(uncond_logits)
            uncond_total_prob = np.exp(uncond_total_logit)
        else:
            uncond_correct_prob = None
            uncond_total_prob = None
            uncond_correct_prob_per_token = None
            uncond_correct_prob_per_char = None
            uncond_correct_prob_per_byte = None

        if incorrect_probs and not np.isnan(correct_prob - np.max(incorrect_probs)):
            margin = correct_prob - np.max(incorrect_probs)
            margin_per_token = correct_prob_per_token - np.max(
                incorrect_probs_per_token
            )
            margin_per_char = correct_prob_per_char - np.max(incorrect_probs_per_char)
            # margin_per_byte = correct_prob_per_byte - np.max(incorrect_probs_per_byte)
            assert -1 <= margin <= 1, f"Margin out of bounds: {margin}"
            assert (
                -1 <= margin_per_token <= 1
            ), f"Margin per token out of bounds: {margin_per_token}"
            assert (
                -1 <= margin_per_char <= 1
            ), f"Margin per char out of bounds: {margin_per_char}"
        else:
            margin = None
            margin_per_token = None
            margin_per_char = None
            # margin_per_byte = None

        # Compute total_logit and total_prob using log-sum-exp trick
        logits = np.array([option["sum_logits"] for option in model_outputs])
        total_logit = log_sum_exp(logits)
        total_prob = np.exp(total_logit)

        logits_per_token = np.array(
            [option["logits_per_token"] for option in model_outputs]
        )
        total_logit_per_token = log_sum_exp(logits_per_token)
        total_prob_per_token = np.exp(total_logit_per_token)

        logits_per_char = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        total_logit_per_char = log_sum_exp(logits_per_char)
        total_prob_per_char = np.exp(total_logit_per_char)

        logits_per_byte = np.array(
            [option["logits_per_char"] for option in model_outputs]
        )
        # total_logit_per_byte = log_sum_exp(logits_per_byte)
        # total_prob_per_byte = np.exp(total_logit_per_byte)

        norm_correct_prob = np.exp(correct_logit - total_logit)
        norm_correct_prob_per_token = np.exp(
            correct_logit_per_token - total_logit_per_token
        )
        norm_correct_prob_per_char = np.exp(
            correct_logit_per_char - total_logit_per_char
        )
        # norm_correct_prob_per_byte = np.exp(
        #     correct_logit_per_byte - total_logit_per_byte
        # )

        if not np.isnan(total_prob):
            assert (
                0 <= total_prob <= len(model_outputs)
            ), f"Total probability out of bounds ({len(model_outputs)}): {total_prob}"
            assert (
                0 <= norm_correct_prob <= 1
            ), f"Normalized correct probability out of bounds: {norm_correct_prob}"
            assert (
                0 <= norm_correct_prob_per_token <= 1
            ), f"Normalized correct probability per token out of bounds: {norm_correct_prob_per_token}"
            assert (
                0 <= norm_correct_prob_per_char <= 1
            ), f"Normalized correct probability per char out of bounds: {norm_correct_prob_per_char}"

            # Checks for inf, -inf, and NaNs
            check_finite_and_nan(total_prob, "total_prob")
            check_finite_and_nan(total_prob_per_token, "total_prob_per_token")
            check_finite_and_nan(total_prob_per_char, "total_prob_per_char")
            check_finite_and_nan(norm_correct_prob, "norm_correct_prob")
            check_finite_and_nan(
                norm_correct_prob_per_token, "norm_correct_prob_per_token"
            )
            check_finite_and_nan(
                norm_correct_prob_per_char, "norm_correct_prob_per_char"
            )

        row_dict = {
            "correct_logit": correct_logit,
            "correct_logit_per_token": correct_logit_per_token,
            "correct_logit_per_char": correct_logit_per_char,
            # "correct_logit_per_byte": correct_logit_per_byte,
            "correct_prob": correct_prob,
            "correct_prob_per_token": correct_prob_per_token,
            "correct_prob_per_char": correct_prob_per_char,
            # "correct_prob_per_byte": correct_prob_per_byte,
            "margin": margin,
            "margin_per_token": margin_per_token,
            "margin_per_char": margin_per_char,
            # "margin_per_byte": margin_per_byte,
            "total_prob": total_prob,
            "total_prob_per_token": total_prob_per_token,
            "total_prob_per_char": total_prob_per_char,
            # "total_prob_per_byte": total_prob_per_byte,
            "uncond_correct_prob": uncond_correct_prob,
            "uncond_correct_prob_per_token": uncond_correct_prob_per_token,
            "uncond_correct_prob_per_char": uncond_correct_prob_per_char,
            # "uncond_correct_prob_per_byte": uncond_correct_prob_per_byte,
            "uncond_total_prob": uncond_total_prob,
            "norm_correct_prob": norm_correct_prob,
            "norm_correct_prob_per_token": norm_correct_prob_per_token,
            "norm_correct_prob_per_char": norm_correct_prob_per_char,
            # "norm_correct_prob_per_byte": norm_correct_prob_per_byte,
        }
        metrics.update(row_dict)

    # 2. Generation tasks
    elif any(key in metrics for key in GENERATE_PRIMARY_METRICS):

        # Case: Codex - Check if model_outputs has 2 elements
        if len(model_outputs) == 2:
            model_outputs = model_outputs[:1]  # pass_at_1

        if len(model_outputs) > 1:
            raise ValueError(
                "Assume generation tasks only have one output (greedy): ",
                len(model_outputs),
            )

        logits = model_outputs[0]["sum_logits"]
        num_tokens = (
            model_outputs[0]["num_tokens"] if model_outputs[0]["num_tokens"] > 0 else 1
        )
        num_chars = (
            len(model_outputs[0]["continuation"])
            if model_outputs[0]["continuation"]
            else 1
        )

        logit_per_token = logits / num_tokens
        logit_per_char = logits / num_chars

        # Case: sum_scores only available in latest version
        if "sum_scores" in model_outputs[0]:
            scores = model_outputs[0]["sum_scores"]
            score_per_token = scores / num_tokens
            score_per_char = scores / num_chars
            check_finite_and_nan(logits, "logit")
            check_finite_and_nan(logit_per_token, "logit_per_token")
            check_finite_and_nan(logit_per_char, "logit_per_char")
        else:
            scores = None
            score_per_token = None
            score_per_char = None

        row_dict = {
            "logit": logits,
            "logit_per_token": logit_per_token,
            "logit_per_char": logit_per_char,
            "score": scores,
            "score_per_token": score_per_token,
            "score_per_char": score_per_char,
        }
        metrics.update(row_dict)

    return metrics


def main():
    args = _parser.parse_args()
    args_dict = vars(args)

    models = args_dict.get("models", "").split(",") if args_dict.get("models", "") != "" else []
    revisions = args_dict.get("revisions", "").split(",") if args_dict.get("revisions", "") != "" else ["main"] * len(models)
    for i, maybe_model in enumerate(models):
        if ":" in maybe_model:
            [models[i], revisions[i]] = maybe_model.split(":", 1)
    model_paths = set(args_dict.get("model_paths", "").split(",")) if args_dict.get("model_paths", "") != "" else set()

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
                    model_paths.add(root)
                # elif os.path.isfile(os.path.join(root, "wandb/latest-run/files/output.log")):
                elif os.path.isfile(os.path.join(root, "COMPLETED")):
                    model_paths.add(root)
        assert not (len(models) > 0 and len(model_paths) > 0), "Cannot specify both --models and --model-paths!"
        print(models, revisions, model_paths)

    ppl_df = pd.read_parquet("hf://datasets/allenai/DataDecide-ppl-results/data/train-00000-of-00001.parquet")
    eval_df = pd.read_parquet("hf://datasets/allenai/DataDecide-eval-results/data/macro_avg-00000-of-00001.parquet")
    num_ft_tasks = len(FT_TASKS)

    for model_path in model_paths:
        print(f"Processing model at {model_path}...")
        model_args_dict = copy.deepcopy(args_dict)
        model_args_dict["model_path"] = model_path.strip()
        # if model_args_dict.get("output_dir", None) is None:
        #     model_args_dict["output_dir"] = os.path.join(
        #         model_args_dict["model_path"], "eval_results"
        #     )
        # else:
        #     model_args_dict["output_dir"] = os.path.join(
        #         model_args_dict["output_dir"],
        #         model_args_dict["model_path"].replace("/", "_"),
        #     )

        path_parts = reversed(pathlib.Path(model_path).parts)
        run_id = ""
        for p in path_parts:
            if "25" in p:
                run_id = p
                break

        wandb_run_path = f"{args_dict['wandb_entity']}/{args_dict['wandb_project']}/runs/{run_id}"
        try:
            wandb_run = wandb.Api().run(wandb_run_path)
        except wandb.errors.CommError as e:
            print(f"Could not find wandb run at {wandb_run_path}, skipping...")
            continue
        # if key not in wandb_run.config or args_dict["wandb_override"]:
        
        model_info = {}
        config = {}

        command = "dpo_tune_cache" if "dpo_tune_cache" in run_id else "finetune" 


        if command == 'finetune':
            model_size_name = None
            try:
                if "lr" in run_id:
                    run_id_part = run_id.split("lr")[0].strip("-").strip("_")
                    run_id_part = "DD-" + run_id_part.rsplit("DD-")[-1]
                elif "learning_rate" in run_id:
                    run_id_parts = run_id.split("_")
                    run_id_part = (
                        run_id_parts[-3] + "_" + run_id_parts[-2] 
                        if "dclm" in run_id_parts[-3] or "dolma" in run_id_parts[-3]
                        else run_id_parts[-2] 
                    )
                else:
                    # import pdb; pdb.set_trace()
                    run_id_part = "DD-" + run_id.rsplit("DD-")[-1]
                _, data, model_size_name, step, seed = run_id_part.split("-")
                model_info = DD_MODEL_SIZES_INFO[model_size_name]
                config["model_size"] = int(model_info["model_size"])
                config["pretrain_steps"] = int(step)
                seed = DD_MODEL_SEEDS[int(seed)]
                pretrain_data = run_id.rsplit(model_size_name, 1)[0].strip("-").rsplit("DD-")[-1].rsplit("dd_")[-1].rsplit("DataDecide-")[-1].strip("-").strip("_")

            except:
                for msz in DD_MODEL_SIZES_INFO:
                    if f"-{msz}_" in run_id:
                        model_info = DD_MODEL_SIZES_INFO[msz]
                        config["model_size"] = int(model_info["model_size"])
                        model_size_name = msz
                if not config.get("model_size"):
                    continue
                # revision = run_id.split(f"-{model_size_name}_")[1].split("_")[0]
                revision = run_id.split(f"{model_size_name}")[1].split("_")[0]
                if "step" in revision:
                    step_str, seed = revision.split("step")[1].split("-")
                    config["pretrain_steps"] = int(step_str.replace("step", ""))
                else:
                    config["pretrain_steps"] = DD_MODEL_SIZES_INFO[model_size_name].get("training_steps")
                    seed = "default"
            
                pretrain_data = run_id.rsplit(f"-{model_size_name}_", 1)[0].strip("-").rsplit("DD-")[-1].rsplit("dd_")[-1].rsplit("DataDecide-")[-1].strip("-").strip("_")
            config["pretrain_tokens"] = config["pretrain_steps"] * DD_SEQ_LEN * model_info.get("batch_size", 1)
            config["pretrain_sequences"] = config["pretrain_steps"] * model_info.get("batch_size", 1)
            config["pretrain_compute"] = 6 * config["model_size"]  * config["pretrain_steps"] * DD_SEQ_LEN * model_info.get("batch_size", 1)  # in A100-80GB-GPU days
            try:
                ppl_df_model_row = ppl_df.query(f"params == '{model_size_name}' and step == {config['pretrain_steps']} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
            except:
                import pdb;pdb.set_trace()
            if len(ppl_df_model_row) != 1:
                print(f"Expected exactly one row for {model_size_name} at step {config['pretrain_steps']} with seed {seed}, but got {len(ppl_df_model_row)}")
            else:
                ppl_df_model_row = ppl_df_model_row.iloc[0]
                for k in ppl_df_model_row.index:
                    if k.startswith("eval/"):
                        config[f"pretrain_{k}"] = ppl_df_model_row[k]
            eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and step == {config['pretrain_steps']} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
            if len(eval_ppl_df_model_rows) != num_ft_tasks:
                # import pdb; pdb.set_trace()
                print(f"Expected exactly {num_ft_tasks} rows for {pretrain_data} {model_size_name} at step {config['pretrain_steps']} with seed {seed}, but got {len(eval_ppl_df_model_rows)}")
                try:
                    eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
                    eval_ppl_df_model_rows["diff"] = abs(eval_ppl_df_model_rows["step"] - config["pretrain_steps"])
                    new_step = eval_ppl_df_model_rows.sort_values(by=["diff"]).iloc[0].step
                    eval_ppl_df_model_rows = eval_df.query(f"params == '{model_size_name}' and step == {new_step} and seed == '{seed}' and data == '{DD_TRAIN_SETS[DD_TRAIN_SETS_SHORT_NAMES_REV.get(pretrain_data, pretrain_data)]}'")
                    print(f"Using step {new_step} instead")
                except:
                    print("Could not find any other step, skipping...")
            for row in eval_ppl_df_model_rows.itertuples():
                task = row.task
                metrics = json.loads(row.metrics)
                for k, v in metrics.items():
                    config[f"pretrain_eval_{task}_{k}"] = v
                try:
                    config["finetune_unique_sequences"] = wandb_run.config["max_train_samples"] if wandb_run.config.get("max_train_samples") else OPEN_INSTRUCT_COMMANDS.get(command, {}).get("sequences")
                    config["finetune_sequences"] = config["finetune_unique_sequences"] * int(wandb_run.config.get("num_train_epochs", 1))
                    # config["finetune_compute"] = 6 * config["model_size"] * config["finetune_sequences"] * DD_SEQ_LEN # in A100-80GB-GPU days
                    config["total_compute_est"] = config["pretrain_compute"] + 6 * config["model_size"] * config["finetune_sequences"] * DD_SEQ_LEN # in A100-80GB-GPU days
                except:
                    import pdb; pdb.set_trace()
            
            # if run_id == '250910-073354_test_match_10M_finetune_100Mtx1_DD-dclm_25_d17_75-20M-5000-2_lr=5e-06':
            #     import pdb; pdb.set_trace()
        elif command == 'dpo_tune_cache':

            try:
                config["dpo_unique_sequences"] = wandb_run.config["max_train_samples"] if wandb_run.config.get("max_train_samples") else OPEN_INSTRUCT_COMMANDS.get(command, {}).get("sequences")
                config["dpo_sequences"] = config["dpo_unique_sequences"] * int(wandb_run.config.get("num_train_epochs", 1))
                ft_run_id = "25" + run_id.split("_25", 1)[1] if "_25" in run_id else "2025" + run_id.split("_2025", 1)[1]
                print(ft_run_id)
                ft_wandb_run_path = f"{args_dict['wandb_entity']}/{args_dict['wandb_project']}/runs/{ft_run_id}"
                ft_wandb_run = wandb.Api().run(ft_wandb_run_path)
                ft_wandb_run_config = ft_wandb_run.summary._json_dict
                for metric in ft_wandb_run_config.keys():
                    if metric.startswith("oe_eval_metrics/"):
                        config[f"finetune_{metric}"] = ft_wandb_run_config.get(metric)
                    elif metric.startswith("pretrain_eval_"):
                        config[f"{metric}"] = ft_wandb_run_config.get(metric)
                
            except:
                if not run_id.startswith("2025_08_2"):
                    import pdb; pdb.set_trace()
        # possible_metrics = [
        #     "primary_metric",
        #     "acc_raw",
        #     "exact_match",
        #     "f1",
        #     "mc1",
        #     "pass_at_1",
        #     "prompt_level_loose_acc",
        #     "maj_at_1",
        # ]
        if os.path.exists(os.path.join(
                model_args_dict["model_path"], "eval_results"
            )):
            metrics_file = os.path.join(
                    model_args_dict["model_path"], "eval_results", "metrics-all.jsonl"
                )
            if not os.path.isfile(metrics_file):
                print(f"Could not find metrics file at {metrics_file}, skipping...")
                continue
            with open(metrics_file, 'r') as f:
                tasks = [json.loads(line) for line in f]
                task_names = [(t["task_idx"], t["task_name"]) for t in tasks]
            # i = 0
            mean_metrics = {}
            done_tasks = set()
            for ti, task in task_names:
                if "::" in task or task in done_tasks:
                    continue
                done_tasks.add(task)
                mean_metrics[task] = {}
                preds_file = os.path.join(
                    model_args_dict["model_path"], "eval_results", f"task-{ti:03}-{task}-predictions.jsonl")
                if not os.path.isfile(preds_file):
                    print(f"Could not find preds file at {preds_file}, skipping...")
                    continue
                with open(preds_file, 'r') as f:
                    predictions = [json.loads(line) for line in f]
                # rows = []
                rows_list = []
                for prediction in predictions:
                    rows_list.append(process_predictions_cheap_decisions(prediction))

                # if task not in PRIMARY_METRICS_OLMES:
                #     raise RuntimeError(f'Could not find "{task}" on path {result_file}!')
                # primary_metric = PRIMARY_METRICS_OLMES[task]
                aggregated_metrics = {}
                for mrow in rows_list:
                    if "em" in mrow:
                        mrow["exact_match"] = mrow.pop("em")
                    # if primary_metric is None:
                    #     for metric in possible_metrics:
                    #         if metric in mrow:
                    #             # Set name forprimary_metric
                    #             primary_metric = metric
                    #             break
                    # if primary_metric is None:
                    #     print(f"Skipping task {task} due to missing primary metric: {mrow}")
                    #     continue

                    # mrow["primary_metric"] = mrow[primary_metric]
                    # mrow["acc_raw"] = mrow["acc_raw"]
                    # mrow["acc_per_char"] = mrow["acc_per_char"]
                    # mrow["acc_per_token"] = mrow["acc_per_token"]
                    # mrow["acc_uncond"] = mrow["acc_uncond"]
                    for key, value in mrow.items():
                        if value is None or isinstance(value, str):
                            continue
                        if key in aggregated_metrics:
                            aggregated_metrics[key].append(value)
                        else:
                            aggregated_metrics[key] = [value]
                # mean_metrics = {k: np.mean(v) for k, v in aggregated_metrics.items()}
                
                for key, values in aggregated_metrics.items():
                    try:
                        arr = np.array(values)
                        try:
                            mean_metrics[task][key] = np.mean(arr)
                        except Exception as e:
                            print(f'Couldnt divide on {key}: {arr}')
                            mean_metrics[task][key] = 0
                    except Exception as e:
                        print(f'Couldnt convert to np array on {key}: {values}')
                        mean_metrics[task][key] = 0
                # i += 1
                
            # task_suites = [t for t in tasks if "::" in t]
            # for task_suite in task_suites:

            for task_suite, task_group_task_names in TASKS.items():
                # import pdb; pdb.set_trace()
                if (None, task_suite) in task_names and all(t in done_tasks for t in task_group_task_names):
                    mean_metrics[task_suite] = {}
                    task_group_tasks = {t["task_name"]: t for t in tasks if t["task_name"] in task_group_task_names}
                    metric_keys = mean_metrics[task_group_task_names[0]].keys()
                    for metric_key in metric_keys:
                        if any(metric_key not in mean_metrics[t] for t in task_group_task_names):
                            continue 
                        sample_val = mean_metrics[task_group_task_names[0]][metric_key]
                        if not isinstance(sample_val, (int, float, bool)):
                            continue
                        # Hack for aggregating metrics starting with "total_"
                        if metric_key.startswith("total_"):
                            mean_metrics[task_suite][metric_key] = sum(
                                mean_metrics[t][metric_key] for t in task_group_task_names
                            )
                            continue

                        mean_metrics[task_suite][metric_key + "_micro"] = sum(
                            mean_metrics[t][metric_key] * task_group_tasks[t]["num_instances"] for t in task_group_tasks
                        ) / sum(task_group_tasks[t]["num_instances"] for t in task_group_tasks)
                        mean_metrics[task_suite][metric_key + "_macro"] = sum(
                            mean_metrics[t][metric_key] for t in task_group_tasks
                        ) / len(task_group_tasks)
            
            for task in mean_metrics:
                for k, v in mean_metrics[task].items():
                    k = f"oe_eval_metrics/{task}/{k}"
                    if k not in wandb_run.summary or args_dict["wandb_override"]:
                        wandb_run.summary[k] = v
        
        try:
            with open(os.path.join(model_args_dict["model_path"], "config.json"), 'r') as f:
                model_config = json.load(f)
                for k, v in model_config.items():
                    if k not in wandb_run.config or not wandb_run.config[k] or args_dict["wandb_override"]:
                        wandb_run.config[k] = v
        except:
            pass
        for k, v in config.items():
            if k not in wandb_run.config or args_dict["wandb_override"]:
                wandb_run.config[k] = v
        wandb_run.update()
        # if "final" in model_path:
        if "step" not in model_path and "epoch" not in model_path and 'final' in model_path: #job is done
            # import pdb;pdb.set_trace()
            if wandb_run.state != "finished": # and os.path.exists(os.path.join(model_path, "COMPLETED")):
                # wandb_run.finish()
                run = wandb.init(project='ft-scaling', id=run_id, resume="must")
                run.finish(exit_code=0)
        elif not "final" in model_path and not has_file_been_modified_recently(
            os.path.join(*pathlib.Path(model_path).parts[:-2],"stderr"), 
            recent_threshold_seconds=60*30
        ) and wandb_run.state == "finished": #job isn't running
            # print(model_path)
            # import pdb; pdb.set_trace()
            run = wandb.init(project='ft-scaling', id=run_id, resume="must")
            run.finish(exit_code=1)
        print(f"Logged metrics to {wandb_run.url}")

if __name__ == "__main__":
    main()
