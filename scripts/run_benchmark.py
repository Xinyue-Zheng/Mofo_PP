# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, NoReturn

import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend


sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


def str_to_bool(value: str) -> bool:
    """
    Converts a string to a boolean: True for 'True', '1', or 'T'; False for 'False', '0', or 'F'.
    """
    if value.lower() in ['true', '1', 't']:
        return True
    elif value.lower() in ['false', '0', 'f']:
        return False
    else:
        raise ValueError("Invalid boolean value. Please enter 'True' or 'False'.")


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    # Base defaults — identical to original MoFo behaviour
    hyper_params = {
        "batch_size": 16,
        "d_model": 24,
        "horizon": 96,
        "lr": 0.01,
        "norm": True,
        "seq_len": 336,
        "patience": 10,
        "periodic": 24,
        "bias": 1,
        "cias": 1,
    }

    # Merge user-supplied --model-hyper-params (JSON) on top of defaults
    raw = args.model_hyper_params
    if isinstance(raw, list):
        raw = " ".join(raw)
    if isinstance(raw, str):
        try:
            hyper_params.update(json.loads(raw))
        except json.JSONDecodeError:
            pass

    # MoFo++: inject CLI flags into hyper-params
    if getattr(args, 'adaptive_period', False):
        hyper_params['adaptive_period'] = True
    if getattr(args, 'channel_attn', False):
        hyper_params['channel_attn'] = True
    if getattr(args, 'n_heads_channel', None) is not None:
        hyper_params['n_heads_channel'] = args.n_heads_channel

    model_config["models"] = {
        "model_hyper_params": hyper_params,
        "adapter": "MoFo_adapter",
        "model_name": "time_series_library.MoFo",
    }

    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]
    evaluation_config["save_path"] = args.save_path

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed
    if args.save_true_pred is not None:
        default_strategy_args["save_true_pred"] = args.save_true_pred
    default_strategy_args["deterministic"] = args.deterministic

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=
        argparse.ArgumentDefaultsHelpFormatter,
    )
    
    
    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        default="rolling_forecast_config.json",
        help="Evaluation config file path",
    )

    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=["dataset/weather.csv"],
        help="List of series names entered by the user",
    )

    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user,"
             "only takes effect when data_name_list is not specified",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default="MoFo_adapter",
        help="Adapter used to adapt the method to our pipeline",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        default = "time_series_library.MoFo",
        help="The relative path of the model that needs to be evaluated",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default='{"batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01, "norm": true, "seq_len": 336, "patience": 10, "periodic": 24, "bias": 1, "cias": 1}',
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # evaluation_config
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )
    parser.add_argument(
        "--strategy-args",
        type=str,
        default='{"horizon": 96}',
        help="Parameters required for evaluating strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
             "by default, use the seed value in the config file"
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="efficient",
        choices=["full", "efficient", "none"],
        help="Specify the type of deterministic behavior for the algorithm. Options are: "
             "'full': Enables full deterministic mode. "
             "'efficient': Fixes only some seeds for efficiency. "
             "'none': No deterministic behavior is applied."
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=0,
        help="List of gpu devices to use, only available in ray backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Time limit for each evaluation task, in seconds",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )

    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="save_path_wheather",
        help="The relative path for saving evaluation results, relative to the result folder",
    )

    parser.add_argument(
        "--save-true-pred",
        type=str_to_bool,
        default=None,
        help="If true, saves the model's prediction results "
             "and the true values in evaluation result file",
    )

    # MoFo++ arguments
    parser.add_argument(
        "--adaptive-period",
        action="store_true",
        default=False,
        help="[MoFo++] Estimate a channel-specific period via FFT instead of using a "
             "fixed global period.",
    )
    parser.add_argument(
        "--channel-attn",
        action="store_true",
        default=False,
        help="[MoFo++] Add a cross-channel attention module after per-channel temporal "
             "processing.",
    )
    parser.add_argument(
        "--n-heads-channel",
        type=int,
        default=4,
        help="[MoFo++] Number of attention heads in the cross-channel attention module.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(
            data_config,
            model_config,
            evaluation_config,
        )

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = "test_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)
