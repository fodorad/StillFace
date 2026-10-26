"""Versioned, strictly validated TOML configuration for feature extractors."""

import tomllib
import types
import typing
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """The configuration file is invalid."""


def load_config[T](path: Path, cls: type[T], supported_version: int) -> T:
    """Load a configuration dataclass from a TOML file.

    Unknown keys, wrong types and a missing or unsupported ``version`` are errors; absent
    keys take the dataclass defaults. Supported field types are ``int``, ``float``, ``str``,
    ``bool``, nested dataclasses, ``tuple[float, ...]``, ``dict[str, X]`` and ``X | None``.

    Args:
        path: TOML file.
        cls: Configuration dataclass to build.
        supported_version: The only accepted value of the ``version`` key.

    Returns:
        The configuration instance.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigError: If the content is invalid.
    """
    data = tomllib.loads(path.read_text())
    if data.get("version") != supported_version:
        raise ConfigError(
            f"{path}: unsupported or missing version {data.get('version')!r}, "
            f"expected {supported_version}"
        )
    return _build(data, cls, "")


def _build[T](table: Any, cls: type[T], where: str) -> T:
    if not isinstance(table, dict):
        raise ConfigError(f"{where or 'config'} must be a table")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(typing.cast(Any, cls))}
    for key in table:
        if key not in known:
            raise ConfigError(f"unknown key '{where}{key}'")
    kwargs = {key: _coerce(value, hints[key], f"{where}{key}") for key, value in table.items()}
    return cls(**kwargs)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(value, options[0], where)
    if is_dataclass(hint):
        return _build(value, hint, f"{where}.")
    if origin is dict:
        _, value_hint = typing.get_args(hint)
        if not isinstance(value, dict):
            raise ConfigError(f"'{where}' must be a table")
        return {k: _coerce(v, value_hint, f"{where}.{k}") for k, v in value.items()}
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"'{where}' must be an array")
        item_hints = [args[0]] * len(value) if args[-1] is Ellipsis else list(args)
        if len(item_hints) != len(value):
            raise ConfigError(f"'{where}' must have {len(item_hints)} items")
        return tuple(_coerce(v, h, where) for v, h in zip(value, item_hints, strict=True))
    if hint is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if hint in (int, str, bool) and type(value) is hint:
        return value
    raise ConfigError(f"'{where}' must be {getattr(hint, '__name__', hint)}")
