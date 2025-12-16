# This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import argparse
from pathlib import Path


def main():
    from msserviceprofiler.ms_service_profiler_ext import compare, split, analyze
    from msserviceprofiler.msservice_advisor import advisor
    from msserviceprofiler.modelevalstate.train import source_to_train
    from msserviceprofiler.modelevalstate.optimizer import optimizer
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="[MindStudio] msserviceprofiler command line tool"
    )
    subparsers = parser.add_subparsers(help="sub-command help")

    source_to_train.arg_parse(subparsers)
    optimizer.arg_parse(subparsers)

    advisor.arg_parse(subparsers)

    analyze.arg_parse(subparsers)
    split.arg_parse(subparsers)
    compare.arg_parse(subparsers)
    args = parser.parse_args()

    # run
    if hasattr(args, "func"):
        args.func(args=args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()