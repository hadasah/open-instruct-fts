# isort: on
import os
import argparse
from accelerate.logging import get_logger

from open_instruct.utils import (
    ArgumentParserPlus,
)
from open_instruct.finetune import FlatArguments, main
from open_instruct.dataset_transformation import TokenizerConfig

from constants import *
logger = get_logger(__name__)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description="Launch a single training job with the specified arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # cmdline_parser.add_argument()
    cmdline_args, fts_args = cmdline_parser.parse_known_args()

    # fts_args = MODEL_HP_DEFAULTS["all"] | MODEL_HP_DEFAULTS["llama"] | cmdline_args.__dict__ | dict(
    #     **{k: v for k, v in PROJECT_SPECS["margsli"].items() if k not in ["MODEL", "DATA_DIR", "NAME_KEYS"]}
    # )
    fts_args = " ".join(f"{k} {v}" for k, v in MODEL_HP_DEFAULTS["all"].items())

    fts_parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = fts_parser.parse_args_into_dataclasses(fts_args.split())
    print(args, tc)
    main(args, tc)
