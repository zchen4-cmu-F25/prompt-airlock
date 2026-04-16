from __future__ import annotations

import copy
import os
from typing import Any

MARIAN_REPO_IDS_DEFAULT: tuple[str, ...] = (
    "Helsinki-NLP/opus-mt-en-fr",
    "Helsinki-NLP/opus-mt-fr-en",
    "Helsinki-NLP/opus-mt-en-de",
    "Helsinki-NLP/opus-mt-de-en",
)

_CONFIG_ENV = "PROMPT_AIRLOCK_CONFIG"
_MODEL_ENV_PREFIX = "PROMPT_AIRLOCK_"
_MODEL_ENV_SUFFIX_MODEL = "_MODEL_PATH"
_MODEL_ENV_SUFFIX_TOKENIZER = "_TOKENIZER_PATH"
_MODEL_ENV_SUFFIX_CONV = "_CONVERSATION_TEMPLATE"
_MARIAN_ENV = "PROMPT_AIRLOCK_MARIAN_REPOS"

_BASE_MODELS: dict[str, dict[str, Any]] = {
    "llama2": {
        "model_path": "/shared_data0/arobey1/llama-2-7b-chat-hf",
        "tokenizer_path": "/shared_data0/arobey1/llama-2-7b-chat-hf",
        "conversation_template": "llama-2",
    },
    "vicuna": {
        "model_path": "/shared_data0/arobey1/vicuna-13b-v1.5",
        "tokenizer_path": "/shared_data0/arobey1/vicuna-13b-v1.5",
        "conversation_template": "vicuna",
    },
    "mistral": {
        "model_path": r"D:\models\Mistral-7B-Instruct-v0.2",
        "tokenizer_path": r"D:\models\Mistral-7B-Instruct-v0.2",
        "conversation_template": "mistral",
    },
    "tinyllama": {
        "model_path": "models/tinyllama",
        "tokenizer_path": "models/tinyllama",
        "conversation_template": "zephyr",
    },
    "llama3.2-3b": {
        "model_path": "models/llama3.2-3b",
        "tokenizer_path": "models/llama3.2-3b",
        "conversation_template": "llama-3",
    },
    "qwen2.5-3b": {
        "model_path": "models/qwen2.5-3b",
        "tokenizer_path": "models/qwen2.5-3b",
        "conversation_template": "qwen-7b-chat",
    },
}


def _deep_merge(
    base: dict[str, Any], overlay: dict[str, Any]
) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as err:
        raise RuntimeError(
            "PyYAML is required to load PROMPT_AIRLOCK_CONFIG; "
            "install pyyaml or unset PROMPT_AIRLOCK_CONFIG."
        ) from err
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_env_model_overrides(models: dict[str, dict[str, Any]]) -> None:
    for model_id in list(models.keys()):
        uid = model_id.upper().replace("-", "_").replace(".", "_")
        mp = os.environ.get(f"{_MODEL_ENV_PREFIX}{uid}{_MODEL_ENV_SUFFIX_MODEL}")
        tp = os.environ.get(f"{_MODEL_ENV_PREFIX}{uid}{_MODEL_ENV_SUFFIX_TOKENIZER}")
        ct = os.environ.get(f"{_MODEL_ENV_PREFIX}{uid}{_MODEL_ENV_SUFFIX_CONV}")
        if mp is not None:
            models[model_id]["model_path"] = mp
        if tp is not None:
            models[model_id]["tokenizer_path"] = tp
        if ct is not None:
            models[model_id]["conversation_template"] = ct


def build_models() -> dict[str, dict[str, Any]]:
    """Assemble model table: defaults -> YAML (PROMPT_AIRLOCK_CONFIG) -> env overrides."""
    models = copy.deepcopy(_BASE_MODELS)
    cfg_path = os.environ.get(_CONFIG_ENV)
    if cfg_path and os.path.isfile(cfg_path):
        yml = _load_yaml_config(cfg_path)
        if "models" in yml and isinstance(yml["models"], dict):
            for name, spec in yml["models"].items():
                if not isinstance(spec, dict):
                    continue
                if name in models:
                    models[name] = _deep_merge(models[name], spec)
                else:
                    models[name] = copy.deepcopy(spec)
    _apply_env_model_overrides(models)
    return models


def build_marian_repo_ids() -> tuple[str, ...]:
    raw = os.environ.get(_MARIAN_ENV)
    if raw and raw.strip():
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    cfg_path = os.environ.get(_CONFIG_ENV)
    if cfg_path and os.path.isfile(cfg_path):
        try:
            yml = _load_yaml_config(cfg_path)
        except OSError:
            return MARIAN_REPO_IDS_DEFAULT
        mr = yml.get("marian_repos")
        if isinstance(mr, list) and mr:
            return tuple(str(x).strip() for x in mr if str(x).strip())
    return MARIAN_REPO_IDS_DEFAULT


_MODELS_CACHE: dict[str, dict[str, Any]] | None = None
_MARIAN_CACHE: tuple[str, ...] | None = None


def get_models() -> dict[str, dict[str, Any]]:
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        _MODELS_CACHE = build_models()
    return _MODELS_CACHE


def get_marian_repo_ids() -> tuple[str, ...]:
    global _MARIAN_CACHE
    if _MARIAN_CACHE is None:
        _MARIAN_CACHE = build_marian_repo_ids()
    return _MARIAN_CACHE


def reset_config_cache() -> None:
    """Clear cached config (e.g. between tests after changing env)."""
    global _MODELS_CACHE, _MARIAN_CACHE, MODELS
    _MODELS_CACHE = None
    _MARIAN_CACHE = None
    MODELS = get_models()


MODELS = get_models()
