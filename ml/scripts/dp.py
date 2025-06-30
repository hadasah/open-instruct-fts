# isort: on
import os
import argparse
from accelerate.logging import get_logger

from open_instruct.utils import (
    ArgumentParserPlus,
)
from open_instruct.dpo_tune_cache import FlatArguments, main
from open_instruct.dataset_transformation import TokenizerConfig

logger = get_logger(__name__)


if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser(
        description="Launch a single training job with the specified arguments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cmdline_parser.add_argument()
    cmdline_args, fts_args = cmdline_parser.parse_known_args()

    fts_parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = fts_parser.parse_args_into_dataclasses(fts_args)
    main(args, tc)
