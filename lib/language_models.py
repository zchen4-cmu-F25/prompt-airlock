from __future__ import annotations

import os
import random
import re
import sys
import time
from typing import List

import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM

import lib.model_configs as model_configs


def configure_hf_hub_env() -> None:
    """Hugging Face Hub defaults; call before transformers/huggingface_hub use."""
    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


configure_hf_hub_env()


def prefetch_hf_snapshots(
    repo_ids: tuple[str, ...] | None = None,
    max_attempts: int = 12,
) -> None:
    """Warm the HF cache with full snapshot downloads to reduce transient HTTP 503 on Marian load."""
    if repo_ids is None:
        repo_ids = model_configs.get_marian_repo_ids()
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


def ensure_nltk_wordnet() -> None:
    import nltk

    for path, name in (
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def synonym_perturb(text: str, rng: random.Random) -> str:
    """Lexical synonym swap via WordNet (fast, no LLM)."""
    if not text.strip():
        return text
    ensure_nltk_wordnet()
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


class MarianPivot:
    """Round-trip translate through a pivot language to paraphrase English text."""

    def __init__(self, en_pivot_model: str, pivot_en_model: str, device: str):
        self._en_pivot_model_name = en_pivot_model
        self._pivot_en_model_name = pivot_en_model
        self.device = device
        self._en_pivot_model = None
        self._en_pivot_tok = None
        self._pivot_en_model = None
        self._pivot_en_tok = None

    def _load_pair(self) -> None:
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


def pick_device(preference: str = "cuda") -> str:
    """Resolve device preference to an actual available device, with MPS fallback."""
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    if preference == "mps" and torch.backends.mps.is_available():
        return "mps"
    if preference == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return preference


def auto_device() -> str:
    """Auto-detect best device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_defense_over_prompts(defense, attack_prompts, desc: str) -> List[bool]:
    from tqdm.auto import tqdm

    jailbroken_results: List[bool] = []
    for prompt in tqdm(attack_prompts, desc=desc):
        output = defense(prompt)
        jb = defense.is_jailbroken(output)
        jailbroken_results.append(jb)
    return jailbroken_results


class LLM:

    """Forward pass through a LLM."""

    def __init__(
        self,
        model_path,
        tokenizer_path,
        conv_template_name,
        device
    ):

        use_cuda = device.startswith('cuda')
        use_mps = device == 'mps'

        _load_kw = dict(
            trust_remote_code=True,
            low_cpu_mem_usage=use_cuda,
            use_cache=True,
        )
        _dtype = torch.float16 if (use_cuda or use_mps) else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=_dtype, **_load_kw
            ).to(device).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=_dtype, **_load_kw
            ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.tokenizer.padding_side = 'left'
        if 'llama-2' in tokenizer_path:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.conv_template = get_conversation_template(
            conv_template_name
        )
        if self.conv_template.name == 'llama-2':
            self.conv_template.sep2 = self.conv_template.sep2.strip()

    def __call__(self, batch, max_new_tokens=100):

        batch_inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=False,
            return_tensors='pt'
        )
        batch_input_ids = batch_inputs['input_ids'].to(self.model.device)
        batch_attention_mask = batch_inputs['attention_mask'].to(self.model.device)

        try:
            outputs = self.model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        except RuntimeError:
            return []

        batch_outputs = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True))
            for i in range(len(batch_input_ids))
        ]
        batch_outputs = [
            output[gen_start_idx[i]:] for i, output in enumerate(batch_outputs)
        ]

        return batch_outputs
