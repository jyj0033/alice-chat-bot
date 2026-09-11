"""配置文件与可编辑敏感配置的存储。"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


# 这些字段从主配置分离到同目录的 secrets.yaml；Web 仍通过合并后的配置读写。
SECRET_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "bot_token",
        "client_secret",
        "client_secret_key",
        "password",
        "secret",
        "secret_key",
        "token",
    }
)
MASKED_SECRET = "********"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取主配置，并覆盖同目录 secrets.yaml 中的敏感字段。"""
    path = Path(config_path)
    config = _read_yaml(path)
    secrets_path = path.with_name("secrets.yaml")
    if secrets_path.exists():
        config = _deep_merge(config, _read_yaml(secrets_path))
    return config


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """保存配置；敏感字段只写入权限为 600 的 secrets.yaml。"""
    path = Path(config_path)
    public, secrets = _split_secrets(config or {})
    _write_yaml(path.with_name("secrets.yaml"), secrets, 0o600)
    _write_yaml(path, public, 0o640)


def mask_secrets(value: Any) -> Any:
    """返回隐藏敏感值的深拷贝，供 Web API 返回。"""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if str(key).lower() in SECRET_KEYS:
                result[key] = MASKED_SECRET if child not in (None, "") else ""
            else:
                result[key] = mask_secrets(child)
        return result
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return deepcopy(value)


def _split_secrets(value: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(value, dict):
        return deepcopy(value), {}

    public = {}
    secrets = {}
    for key, child in value.items():
        if str(key).lower() in SECRET_KEYS:
            if child not in (None, ""):
                secrets[key] = deepcopy(child)
            continue

        public_child, secret_child = _split_secrets(child)
        public[key] = public_child
        if secret_child:
            secrets[key] = secret_child
    return public, secrets


def _write_yaml(path: Path, data: dict[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                data,
                handle,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
        os.chmod(path, mode)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
