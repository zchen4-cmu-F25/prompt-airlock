"""
Semantic Smoothing defense: replace character-level noise with semantic perturbations
(synonym replacement, back-translation, alternate-pivot paraphrase), run the target LLM
on each variation, then majority-vote on jailbreak vs. refusal (same high-level pattern
as SmoothLLM).

Dependencies (install once):
    pip install nltk transformers sentencepiece

First run may download NLTK WordNet data and Helsinki-NLP Marian checkpoints (~hundreds of MB).

Usage (unified entry point recommended):
    python main.py semantic --results_dir ./results_semantic --target_model mistral

Legacy:
    python semantic_smoothing.py --results_dir ./results_semantic --target_model mistral

Compare semantic vs. character-level SmoothLLM (same copy count):
    python main.py semantic --defense both --results_dir ./results_semantic --target_model mistral

See lib/model_configs.py for paths; use PROMPT_AIRLOCK_CONFIG or env overrides.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Callable, List

import numpy as np
import pandas as pd
import torch

import lib.attacks as attacks
import lib.defenses as defenses
import lib.language_models as language_models
import lib.model_configs as model_configs


class SemanticSmoothLLM(defenses.Defense):
    """
    Semantic smoothing: cycle synonym replacement, EN-FR-EN back-translation, and
    EN-DE-EN paraphrase (structural / phrasing change without a separate T5/BERT).
    """

    def __init__(
        self,
        target_model,
        num_copies: int,
        translate_device: str = "cpu",
        seed: int = 0,
    ):
        super().__init__(target_model)
        self.num_copies = num_copies
        self.translate_device = translate_device
        self._rng = random.Random(seed)

        self._bt_fr = language_models.MarianPivot(
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-fr-en",
            translate_device,
        )
        self._bt_de = language_models.MarianPivot(
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-de-en",
            translate_device,
        )

        self._methods: List[Callable[[str], str]] = [
            lambda t: language_models.synonym_perturb(t, self._rng),
            lambda t: self._bt_fr.round_trip(t),
            lambda t: self._bt_de.round_trip(t),
        ]

    @torch.no_grad()
    def __call__(self, prompt, batch_size: int = 64, max_new_len: int = 100):
        del max_new_len

        base_pert = prompt.perturbable_prompt
        base_full = prompt.full_prompt

        all_inputs: List[str] = []
        for i in range(self.num_copies):
            fn = self._methods[i % len(self._methods)]
            try:
                new_pert = fn(base_pert)
            except Exception:
                new_pert = base_pert
            if not new_pert or not str(new_pert).strip():
                new_pert = base_pert
            full = base_full.replace(base_pert, new_pert, 1)
            all_inputs.append(full)

        all_outputs: List[str] = []
        for i in range(self.num_copies // batch_size + 1):
            batch = all_inputs[i * batch_size : (i + 1) * batch_size]
            if not batch:
                continue
            batch_outputs = self.target_model(batch=batch, max_new_tokens=prompt.max_new_tokens)
            all_outputs.extend(batch_outputs)
            torch.cuda.empty_cache()

        are_copies_jailbroken = [self.is_jailbroken(s) for s in all_outputs]
        if not are_copies_jailbroken:
            raise ValueError("LLM did not generate any outputs.")

        jb_percentage = float(np.mean(are_copies_jailbroken))
        smooth_jb = jb_percentage > 0.5
        outputs_and_jbs = zip(all_outputs, are_copies_jailbroken)
        majority_outputs = [out for out, jb in outputs_and_jbs if jb == smooth_jb]
        return random.choice(majority_outputs)


def run_semantic_experiment(args) -> None:
    os.makedirs(args.results_dir, exist_ok=True)

    lm_device = language_models.pick_device(args.llm_device)
    tr_device = language_models.pick_device(args.translate_device)

    config = model_configs.get_models()[args.target_model]
    target_model = language_models.LLM(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
        conv_template_name=config["conversation_template"],
        device=lm_device,
    )

    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model,
    )
    attack_prompts = attack.prompts

    rows: List[dict] = []

    if args.defense in ("semantic", "both"):
        print("Prefetching Marian checkpoints (reduces HTTP 503 during first load)...", flush=True)
        language_models.prefetch_hf_snapshots()

        defense_sem = SemanticSmoothLLM(
            target_model=target_model,
            num_copies=args.num_copies,
            translate_device=tr_device,
            seed=args.seed,
        )
        jb_sem = language_models.run_defense_over_prompts(
            defense_sem, attack_prompts, "semantic smoothing"
        )
        rows.append(
            {
                "Defense": "SemanticSmoothLLM",
                "Copies": args.num_copies,
                "Extra": "synonym+EN-FR-EN+EN-DE-EN",
                "JB percentage": float(np.mean(jb_sem)) * 100.0,
                "Trial index": args.trial,
                "Attack log": args.attack_logfile,
            }
        )

    if args.defense in ("smoothllm", "both"):
        defense_sl = defenses.SmoothLLM(
            target_model=target_model,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.num_copies,
        )
        jb_sl = language_models.run_defense_over_prompts(
            defense_sl, attack_prompts, "SmoothLLM (random)"
        )
        rows.append(
            {
                "Defense": "SmoothLLM",
                "Copies": args.num_copies,
                "Extra": f"{args.smoothllm_pert_type}@{args.smoothllm_pert_pct}%",
                "JB percentage": float(np.mean(jb_sl)) * 100.0,
                "Trial index": args.trial,
                "Attack log": args.attack_logfile,
            }
        )

    summary_df = pd.DataFrame(rows)

    if args.defense == "both":
        out_name = "summary_compare.pd"
    elif args.defense == "semantic":
        out_name = "summary_semantic.pd"
    else:
        out_name = "summary_smoothllm.pd"

    out_path = os.path.join(args.results_dir, out_name)
    summary_df.to_pickle(out_path)
    print(summary_df.to_string(index=False))
    print(f"Wrote {out_path}")


def add_semantic_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--results_dir", type=str, default="./results_semantic")
    p.add_argument("--trial", type=int, default=0)
    p.add_argument(
        "--defense",
        type=str,
        default="semantic",
        choices=["semantic", "smoothllm", "both"],
        help="semantic=SemanticSmoothLLM; smoothllm=random SmoothLLM; both=run both for comparison.",
    )
    p.add_argument(
        "--target_model",
        type=str,
        default="mistral",
        choices=list(model_configs.get_models().keys()),
        help="Target LLM id (see lib/model_configs.py or PROMPT_AIRLOCK_CONFIG).",
    )
    p.add_argument("--attack", type=str, default="GCG", choices=["GCG", "PAIR"])
    p.add_argument(
        "--attack_logfile",
        type=str,
        default="data/GCG/vicuna_behaviors.json",
        help="Vicuna GCG behaviors JSON (goal/target/controls).",
    )
    p.add_argument(
        "--num_copies",
        type=int,
        default=9,
        help="Copies per prompt (semantic cycles + SmoothLLM random copies).",
    )
    p.add_argument(
        "--smoothllm_pert_pct",
        type=int,
        default=10,
        help="SmoothLLM random perturbation percentage (smoothllm / both only).",
    )
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
        "--llm_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for target LLM (default: cuda if available).",
    )
    p.add_argument(
        "--translate_device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Marian back-translation runs here; default cpu to save VRAM.",
    )
    p.add_argument("--seed", type=int, default=0)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Semantic Smoothing evaluation (SmoothLLM-style voting).")
    add_semantic_arguments(p)
    return p


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = build_arg_parser()
    run_semantic_experiment(parser.parse_args())
