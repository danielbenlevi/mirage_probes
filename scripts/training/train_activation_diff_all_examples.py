#!/usr/bin/env python3
"""Train probes on activation-difference features (all examples)."""

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.train_mlp_all_examples as core


def parse_args() -> argparse.Namespace:
    parser = core.build_arg_parser(
        description=(
            "Train logistic-regression probes on all examples using activation differences "
            "(with-image activations minus without-image activations) per feature, "
            "with multi-seed split handling."
        ),
        default_save_dir="./tmp_artifacts/activation_diff_all_examples_probe_results",
        default_cache_path="./tmp_artifacts/all_examples_layer_features_activation_diff.pt",
    )
    parser.add_argument(
        "--use_nonlinear_mlp",
        action="store_true",
        help="Use a non-linear MLP probe instead of logistic regression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core.run_all_examples_experiment(
        args=args,
        script_name="train_activation_diff_all_examples",
        probe_type="mlp" if bool(args.use_nonlinear_mlp) else "logreg",
        feature_variant="activation_diff",
    )


if __name__ == "__main__":
    main()
