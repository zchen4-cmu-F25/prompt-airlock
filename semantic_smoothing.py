"""
Semantic Smoothing defense: replace character-level noise with semantic perturbations
(synonym replacement, back-translation, alternate-pivot paraphrase), run the target LLM
on each variation, then majority-vote on jailbreak vs. refusal (same high-level pattern
as SmoothLLM).

Dependencies (install once):
    pip install nltk transformers sentencepiece

First run may download NLTK WordNet data and Helsinki-NLP Marian checkpoints (~hundreds of MB).

Usage (Vicuna GCG behaviors + local Mistral):
    python semantic_smoothing.py --results_dir ./results_semantic --target_model mistral

Compare semantic vs. character-level SmoothLLM (same copy count):
    python semantic_smoothing.py --defense both --results_dir ./results_semantic --target_model mistral

See lib/model_configs.py (MARK: prompt-airlock) for the Mistral path.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from typing import Callable, List

# Hugging Face Hub: must run before importing transformers / huggingface_hub transitively.
def _configure_hf_hub_env() -> None:
    if sys.platform == "win32":
        # Avoid symlink cache on Windows (no Developer Mode); uses file copies instead of symlinks.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


_configure_hf_hub_env()

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import lib.attacks as attacks
import lib.defenses as defenses
import lib.language_models as language_models
import lib.model_configs as model_configs


MARIAN_REPO_IDS = (
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-de-en",
)


def _prefetch_hf_snapshots(repo_ids: tuple[str, ...], max_attempts: int = 12) -> None:
    """Warm the HF cache with full snapshot downloads to reduce transient HTTP 503 on Marian load."""
    from huggingface_hub import snapshot_download

    for repo_id in repo_ids:
        last_err: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                snapshot_download(repo_id, local_files_only=False)
                break
            except BaseException as err:
                last_err = err
                wait = min(120, 2 ** min(attempt, 6))
                time.sleep(wait)
        else:
            assert last_err is not None
            raise last_err


# --- NLTK (synonym replacement) -------------------------------------------------

def _ensure_nltk_wordnet() -> None:
    import nltk

    for path, name in (
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _synonym_perturb(text: str, rng: random.Random) -> str:
    """Lexical synonym swap via WordNet (fast, no LLM)."""
    if not text.strip():
        return text
    _ensure_nltk_wordnet()
    from nltk.corpus import wordnet as wn

    tokens = text.split()
    if not tokens:
        return text

    indices = list(range(len(tokens)))
    rng.shuffle(indices)
    for i in indices:
        tok = tokens[i]
        m = re.match(r"^([^A-Za-z0-9]*)([A-Za-z][A-Za-z\-']*)([^A-Za-z0-9]*)$", tok)
        if not m:
            continue
        prefix, core, suffix = m.groups()
        if len(core) < 3:
            continue
        lower = core.lower()
        syns: List[str] = []
        for syn in wn.synsets(lower):
            for lm in syn.lemmas():
                w = lm.name().replace("_", " ")
                if w.lower() != lower and w.isascii():
                    syns.append(w)
        if not syns:
            continue
        rep = rng.choice(syns)
        if core[0].isupper():
            rep = rep[:1].upper() + rep[1:] if len(rep) > 1 else rep.upper()
        tokens[i] = prefix + rep + suffix
        return " ".join(tokens)
    return text


# --- Marian back-translation (lightweight seq2seq) ------------------------------

class _MarianPivot:
    """Round-trip translate through a pivot language to paraphrase English text."""

    def __init__(self, en_pivot_model: str, pivot_en_model: str, device: str):
        self._en_pivot_model_name = en_pivot_model
        self._pivot_en_model_name = pivot_en_model
        self.device = device
        self._en_pivot_model = None
        self._en_pivot_tok = None
        self._pivot_en_model = None
        self._pivot_en_tok = None

    def _load_pair(self):
        if self._en_pivot_model is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer

        self._en_pivot_tok = MarianTokenizer.from_pretrained(self._en_pivot_model_name)
        self._en_pivot_model = MarianMTModel.from_pretrained(self._en_pivot_model_name)
        self._en_pivot_model.to(self.device).eval()

        self._pivot_en_tok = MarianTokenizer.from_pretrained(self._pivot_en_model_name)
        self._pivot_en_model = MarianMTModel.from_pretrained(self._pivot_en_model_name)
        self._pivot_en_model.to(self.device).eval()

    @staticmethod
    def _gen(model, tokenizer, text: str, device: str) -> str:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out_ids = model.generate(**enc, max_length=512, num_beams=4, early_stopping=True)
        return tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def round_trip(self, text: str) -> str:
        if not text.strip():
            return text
        self._load_pair()
        mid = self._gen(self._en_pivot_model, self._en_pivot_tok, text, self.device)
        if not mid.strip():
            return text
        back = self._gen(self._pivot_en_model, self._pivot_en_tok, mid, self.device)
        return back.strip() or text


# --- Defense --------------------------------------------------------------------

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

        self._bt_fr = _MarianPivot(
            "Helsinki-NLP/opus-mt-en-fr",
            "Helsinki-NLP/opus-mt-fr-en",
            translate_device,
        )
        self._bt_de = _MarianPivot(
            "Helsinki-NLP/opus-mt-en-de",
            "Helsinki-NLP/opus-mt-de-en",
            translate_device,
        )

        self._methods: List[Callable[[str], str]] = [
            lambda t: _synonym_perturb(t, self._rng),
            lambda t: self._bt_fr.round_trip(t),
            lambda t: self._bt_de.round_trip(t),
        ]

    @torch.no_grad()
    def __call__(self, prompt, batch_size: int = 64, max_new_len: int = 100):
        del max_new_len  # API parity with SmoothLLM; generation uses prompt.max_new_tokens

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


def _pick_device(for_translate: str) -> str:
    if for_translate == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if for_translate == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return for_translate


def _run_defense_loop(defense: defenses.Defense, attack_prompts, desc: str) -> List[bool]:
    jailbroken_results: List[bool] = []
    for prompt in tqdm(attack_prompts, desc=desc):
        output = defense(prompt)
        jb = defense.is_jailbroken(output)
        jailbroken_results.append(jb)
    return jailbroken_results


def main(args):
    os.makedirs(args.results_dir, exist_ok=True)

    lm_device = _pick_device(args.llm_device)
    tr_device = _pick_device(args.translate_device)

    config = model_configs.MODELS[args.target_model]
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
        _prefetch_hf_snapshots(MARIAN_REPO_IDS)

        defense_sem = SemanticSmoothLLM(
            target_model=target_model,
            num_copies=args.num_copies,
            translate_device=tr_device,
            seed=args.seed,
        )
        jb_sem = _run_defense_loop(defense_sem, attack_prompts, "semantic smoothing")
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
        jb_sl = _run_defense_loop(defense_sl, attack_prompts, "SmoothLLM (random)")
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


def build_arg_parser():
    p = argparse.ArgumentParser(description="Semantic Smoothing evaluation (SmoothLLM-style voting).")
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
        choices=list(model_configs.MODELS.keys()),
        help="Use mistral (local path in lib/model_configs.py) for the target LLM.",
    )
    p.add_argument("--attack", type=str, default="GCG", choices=["GCG", "PAIR"])
    # Behaviors: Vicuna GCG log only for this project iteration.
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
        help="Device for Mistral (default: cuda if available).",
    )
    p.add_argument(
        "--translate_device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Marian back-translation runs here; default cpu to save VRAM for Mistral.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = build_arg_parser()
    main(parser.parse_args())
