import argparse
import importlib
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import lib.attacks as attacks
import lib.defenses as defenses
import lib.language_models as language_models
import lib.model_configs as model_configs
import semantic_smoothing


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend ``smoothllm`` so legacy invocations ``python main.py --target_model ...`` still work."""
    if not argv:
        return ["smoothllm"]
    if argv[0] in ("-h", "--help"):
        return argv
    if argv[0] not in ("smoothllm", "semantic", "smoke"):
        return ["smoothllm"] + argv
    return argv


def run_smoothllm(args) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    device = language_models.auto_device()
    config = model_configs.get_models()[args.target_model]
    target_model = language_models.LLM(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
        conv_template_name=config["conversation_template"],
        device=device,
    )

    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies,
        detector=args.smoothllm_detector,
    )

    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model,
    )

    classifications = []
    for i, prompt in tqdm(enumerate(attack.prompts)):
        output = defense(prompt)
        label = defense.classify_output(output)
        classifications.append(label)

    n_total = len(classifications)
    n_refusal = classifications.count('refusal')
    n_confusion = classifications.count('confusion')
    n_jailbreak = classifications.count('jailbreak')

    summary_df = pd.DataFrame.from_dict({
        'Number of smoothing copies': [args.smoothllm_num_copies],
        'Perturbation type': [args.smoothllm_pert_type],
        'Perturbation percentage': [args.smoothllm_pert_pct],
        'Detector': [args.smoothllm_detector],
        'Trial index': [args.trial],
        'Refusal %': [n_refusal / n_total * 100 if n_total else 0],
        'Confusion %': [n_confusion / n_total * 100 if n_total else 0],
        'Jailbreak %': [n_jailbreak / n_total * 100 if n_total else 0],
    })
    summary_df.to_pickle(os.path.join(
        args.results_dir, "summary.pd"
    ))
    print(summary_df.to_string())


def run_smoke(_args) -> None:
    """
    Fast regression check: imports and config resolution (no GPU, no model load).
    Intended for CI: ``python main.py smoke`` after ``pip install -r requirements.txt``.
    """
    del _args
    failures: list[str] = []
    for mod in (
        "lib.model_configs",
        "lib.language_models",
        "lib.attacks",
        "lib.defenses",
        "lib.perturbations",
        "semantic_smoothing",
    ):
        try:
            importlib.import_module(mod)
        except Exception as err:
            failures.append(f"{mod}: {err!r}")

    models = model_configs.get_models()
    if not models:
        failures.append("get_models() returned empty mapping")

    marian = model_configs.get_marian_repo_ids()
    if not marian:
        failures.append("get_marian_repo_ids() returned empty tuple")

    if failures:
        for line in failures:
            print("SMOKE FAIL:", line, file=sys.stderr)
        raise SystemExit(1)
    print("smoke: ok (imports + config); models:", list(models.keys()))


def _add_smoothllm_arguments(p) -> None:
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument(
        "--target_model",
        type=str,
        default="vicuna",
        choices=list(model_configs.get_models().keys()),
    )
    p.add_argument("--attack", type=str, default="GCG", choices=["GCG", "PAIR"])
    p.add_argument(
        "--attack_logfile",
        type=str,
        default="data/GCG/vicuna_behaviors.json",
    )
    p.add_argument("--smoothllm_num_copies", type=int, default=10)
    p.add_argument("--smoothllm_pert_pct", type=int, default=10)
    p.add_argument(
        "--smoothllm_pert_type",
        type=str,
        default="RandomSwapPerturbation",
        choices=[
            "RandomSwapPerturbation",
            "RandomPatchPerturbation",
            "RandomInsertPerturbation",
        ],
    )
    p.add_argument(
        "--smoothllm_detector",
        type=str,
        default="three_class",
        choices=["binary", "three_class"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SmoothLLM + Semantic Smoothing evaluations.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smooth = sub.add_parser("smoothllm", help="Character-level SmoothLLM defense.")
    _add_smoothllm_arguments(p_smooth)

    p_sem = sub.add_parser("semantic", help="Semantic smoothing / SmoothLLM comparison.")
    semantic_smoothing.add_semantic_arguments(p_sem)

    sub.add_parser("smoke", help="Import and config smoke test (CI-friendly, no GPU).")

    return parser


def main() -> None:
    argv = _normalize_argv(sys.argv[1:])
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "smoothllm":
        torch.manual_seed(args.trial)
        random.seed(args.trial)
        np.random.seed(args.trial)
        run_smoothllm(args)
    elif args.cmd == "semantic":
        semantic_smoothing.run_semantic_experiment(args)
    elif args.cmd == "smoke":
        run_smoke(args)
    else:
        parser.error(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
