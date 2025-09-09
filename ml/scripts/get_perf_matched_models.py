import os
import json
import copy
import pandas as pd
pd.options.mode.copy_on_write = True 
from constants import *


# def grab_perf_matched_models(
#         pretrain_data_set,
#         pretrain_model_size,
#         pretrain_model_seed,
#         pretrain_model_step,
#         match_pretrain_data_set,
#         threshold_metric_diff=0.1,
#         matched_models_dir=os.path.join(USER_PROJECT_SPEC["PROJECT_DIR"], "matched_models"),
# ):
#     with open(
#         os.path.join(
#             matched_models_dir, 
#             pretrain_data_set, 
#             pretrain_model_size,
#             pretrain_model_seed,
#             pretrain_model_step,
#             f"perf_matches_ppl_{pretrain_model_size}.json"
#     ), 'r') as f:
#         perf_matches = [
#             {
#                 "model_size": m_size, 
#                 "model_seed": m_seed, 
#                 "model_step": m_seed_dict["matched_step"], 
#                 "matched_score": m_seed_dict["matched_score"], 
#                 "score_diff": m_seed_dict["score_diff"]
#             } 
#             for m_size, m_size_dict in json.load(f)[match_pretrain_data_set].items() 
#             for m_seed, m_seed_dict in m_size_dict.items()
#         ]
#         perf_matches = sorted(perf_matches, key=lambda item: item["score_diff"])
#         perf_match_str = ""
#         i = 0
#         while True:
#             if i >= len(perf_matches) or perf_matches[i]["score_diff"] > threshold_metric_diff:
#                 break
#             perf_match_str += f"{match_pretrain_data_set}:{perf_matches[i]["model_size"]}:{perf_matches[i]["model_step"]}:{perf_matches[i]["model_seed"]}"
#             i += 1
#         print(perf_match_str)


def grab_perf_matched_models(
        matched_dict,
        match_pretrain_data_sets,
        threshold_metric_diff=0.1,
):
    for match_pretrain_data_set in match_pretrain_data_sets:
        perf_matches = []
        for m_size, m_size_dict in matched_dict[match_pretrain_data_set].items():
            if not isinstance(m_size_dict, dict) or len(m_size_dict) == 0:
                continue
            for m_seed, m_seed_dict in m_size_dict.items():
                if not isinstance(m_seed_dict, dict) or len(m_seed_dict) == 0:
                    continue
                if "matched_step" not in m_seed_dict:
                    import pdb;pdb.set_trace()
                perf_matches.append(
                    {
                        "model_size": m_size, 
                        "model_seed": m_seed, 
                        "model_step": m_seed_dict["matched_step"], 
                        "matched_score": m_seed_dict["matched_score"], 
                        "score_diff": m_seed_dict["score_diff"]
                    } 
                )
        
        perf_matches = sorted(perf_matches, key=lambda item: item["score_diff"])
        perf_match_str = []
        i = 0
        while True:
            if i >= len(perf_matches) or perf_matches[i]["score_diff"] > threshold_metric_diff:
                break
            perf_match_str += [f"{match_pretrain_data_set}:{perf_matches[i]['model_size']}:{str(int(perf_matches[i]['model_step']))}:{DD_MODEL_SEEDS_DICT[perf_matches[i]['model_seed']]}"]
            i += 1
        print(match_pretrain_data_set, " ", '::'.join(perf_match_str))

def match_models_by_perf(
        df: pd.DataFrame,
        pretrain_data: str, 
        pretrain_step_portions: list = [1],
        pretrain_steps: list = [],
        match_pretrain_data_sets: list = list(DD_TRAIN_SETS.keys()),
        metric_to_match: str = "ppl", task_to_match: str = "",
        output_dir: str = "./matched_models",
        ):
    
        # for model_size in DD_MODEL_SIZES_INFO:
        #     perf_matches[pretrain_data][model_size] = {}
        #     model_info = DD_MODEL_SIZES_INFO[model_size]
        #     full_pretrain_steps = model_info.get("training_steps", None)
        #     temp_steps = [p * full_pretrain_steps for p in pretrain_step_portions]
        #     for step in temp_steps:
        #         step_rows = df.query(f"params == '{model_size}' and data == '{DD_TRAIN_SETS[pretrain_data]}'")
        #         step_rows['step_diff'] = abs(step_rows['step'] - step)
        #         step_rows = step_rows.sort_values(by=['step_diff'])
        #         pretrain_steps.append(step_rows.iloc[0]['step'])
        #     for seed in DD_MODEL_SEEDS_DICT.keys():
        
        model_perf_matches = {}
        query = f"params == '{model_size}' and seed == '{seed}' and step == {step} and data == '{DD_TRAIN_SETS[pretrain_data]}'"
        if metric_to_match != 'ppl':
            query += f" and task == "
        model_rows = df.query(query)
        if len(model_rows) != 1:
            return {}
            # print(f"Expected exactly one row for {model_size} at step {step} with seed {seed}, but got {len(model_rows)}")
        if metric_to_match == 'ppl':
            model_score = model_rows.iloc[0][f'eval/{task_to_match}/Perplexity']
        else:
            metrics = json.loads(model_rows.iloc[0].metrics)
            if metric_to_match not in metrics.keys():
                print(f"Metric {metric_to_match} not found in metrics for {m_model_size} with seed {m_seed} on {m_pretrain_data}")
                return {}
            model_score = metrics[metric_to_match]
        for m_pretrain_data in match_pretrain_data_sets:
            model_perf_matches[m_pretrain_data] = {}
            for m_model_size in DD_MODEL_SIZES_INFO:
                model_perf_matches[m_pretrain_data][m_model_size] = {}
                for m_seed in DD_MODEL_SEEDS_DICT.keys():
                    query = f"params == '{m_model_size}' and seed == '{m_seed}' and data == '{DD_TRAIN_SETS[m_pretrain_data]}'"
                    if metric_to_match != 'ppl':
                        query += f" and task == '{task_to_match}'"
                    m_df = df.query(query)
                    if len(m_df) == 0:
                        continue
                    if metric_to_match == 'ppl':
                        m_df["diff"] = abs(m_df[f'eval/{task_to_match}/Perplexity'] - model_score)
                        m_df = m_df.sort_values(by=["diff"])
                        best_match_row = m_df.iloc[0]
                        model_perf_matches[m_pretrain_data][m_model_size][m_seed] = {
                            "matched_step": best_match_row.step,
                            "matched_score": best_match_row[f'eval/{task_to_match}/Perplexity'],
                            "score_diff": best_match_row["diff"]
                        }
                    else:
                        metric_keys = json.loads(m_df.iloc[0].metrics).keys()
                        if metric_to_match not in metric_keys:
                            print(f"Metric {metric_to_match} not found in metrics for {m_model_size} with seed {m_seed} on {m_pretrain_data}")
                            continue
                        m_df["diff"] = abs(m_df.metrics.apply(lambda x: json.loads(x)[metric_to_match]) - model_score)
                        m_df = m_df.sort_values(by=["diff"])
                        best_match_row = m_df.iloc[0]
                        model_perf_matches[m_pretrain_data][m_model_size][m_seed] = {
                            "matched_step": best_match_row.step,
                            "matched_score": json.loads(best_match_row.metrics)[metric_to_match],
                            "score_diff": best_match_row["diff"]
                        }
                

    # print(perf_matches)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get performance matched models based on specified metric and task.")
    parser.add_argument("--pretrain_data_sets", nargs='+', default=list(DD_TRAIN_SETS.keys()), help="List of pretraining datasets to consider.")
    parser.add_argument("--pretrain_steps", nargs='+', type=int, default=[], help="List of specific pretraining steps to consider.")
    parser.add_argument("--pretrain_step_portions", nargs='+', type=float, default=[1], help="List of portions of full training steps to consider (e.g., 0.25, 0.5, 1).")
    parser.add_argument("--match_pretrain_data_sets", nargs='+', default=list(DD_TRAIN_SETS.keys()), help="List of pretraining datasets to match against.")
    parser.add_argument("--metric_to_match", type=str, default="ppl", help="Metric to use for matching (e.g., 'ppl', 'accuracy').")
    parser.add_argument("--task_to_match", type=str, default="", help="Task to use for matching when metric is not 'ppl'.")
    parser.add_argument("--output_dir", type=str, default=f"{USER_PROJECT_SPEC['PROJECT_DIR']}/matched_models", help="Directory to save matched model information.")
    parser.add_argument("--print", action="store_true")
    parser.add_argument("--threshold_metric_diff", type=float, default=0.1, help="Threshold for metric difference to consider a match.")
    
    
    args = parser.parse_args()
    
    if args.metric_to_match == 'ppl':
        df = pd.read_parquet("hf://datasets/allenai/DataDecide-ppl-results/data/train-00000-of-00001.parquet")
    else:
        df = pd.read_parquet("hf://datasets/allenai/DataDecide-eval-results/data/macro_avg-00000-of-00001.parquet")
    

    perf_matches = {}

    for pretrain_data in args.pretrain_data_sets:
        # pretrain_data = DD_TRAIN_SETS
        perf_matches[pretrain_data] = {}
        
        for model_size in DD_MODEL_SIZES_INFO:
            perf_matches[pretrain_data][model_size] = {}
            model_info = DD_MODEL_SIZES_INFO[model_size]
            full_pretrain_steps = model_info.get("training_steps", None)
            temp_steps = [p * full_pretrain_steps for p in args.pretrain_step_portions]
            pretrain_steps = copy.copy(args.pretrain_steps)
            for step in temp_steps:
                step_rows = df.query(f"params == '{model_size}' and data == '{DD_TRAIN_SETS[pretrain_data]}'")
                step_rows['step_diff'] = abs(step_rows['step'] - step)
                step_rows = step_rows.sort_values(by=['step_diff'])
                pretrain_steps.append(step_rows.iloc[0]['step'])
            for seed in DD_MODEL_SEEDS_DICT.keys():
                seed_rows = df.query(f"params == '{model_size}' and seed == '{seed}' and step == {step} and data == '{DD_TRAIN_SETS[pretrain_data]}'")
                if len(seed_rows) == 0:
                    continue
                perf_matches[pretrain_data][model_size][seed] = {}
                for step in pretrain_steps:
                    model_seed_dir = os.path.join(args.output_dir, pretrain_data, model_size, seed, str(step))
                    os.makedirs(model_seed_dir, exist_ok=True)
                    fname = os.path.join(model_seed_dir, f"perf_matches_{args.metric_to_match}_{args.task_to_match}.json")
                    if os.path.exists(fname):
                        with open(fname, 'r') as f:
                            perf_matches[pretrain_data][model_size][seed][step] = json.load(f)
                            
                            print(fname) 
                    else:
                        perf_matches[pretrain_data][model_size][seed][step] = match_models_by_perf(
                            df=df,
                            pretrain_data=pretrain_data,
                            match_pretrain_data_sets=args.match_pretrain_data_sets,
                            metric_to_match=args.metric_to_match,
                            task_to_match=args.task_to_match,
                            output_dir=args.output_dir
                        )
                        with open(fname, 'w') as f:
                            json.dump(perf_matches, f, indent=2)
                    
                    if args.print:
                        print(f"Performance matched models for {pretrain_data}, {model_size}, seed {seed}, step {step}:")
                        grab_perf_matched_models(
                            perf_matches[pretrain_data][model_size][seed][step],
                            match_pretrain_data_sets=args.match_pretrain_data_sets, # just print for the first match dataset
                            threshold_metric_diff=args.threshold_metric_diff,
                        )