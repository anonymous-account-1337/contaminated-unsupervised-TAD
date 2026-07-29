import json

from collections.abc import Mapping
from numbers import Real
from statistics import mean, stdev
from typing import Any
from hashlib import sha256


def hash_dict(d, included_keys=None, excluded_keys=None):
    included = set(d.keys()) if included_keys is None else set(included_keys)
    excluded = set() if excluded_keys is None else set(excluded_keys)

    filtered = {
        k: d[k]
        for k in sorted(included)
        if (
            k in d
            and k not in excluded
            and not str(k).startswith('_')
        )
    }

    payload = json.dumps(
        filtered,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )

    return sha256(payload.encode('utf-8')).hexdigest()


def flatten_dict(v, k=None, l=None):
    if l is None:
        l = []

    if isinstance(v, dict):
        for key, val in v.items():
            if key[0] == '_':
                key = key[1:]

            new_key = key if k is None else f'{k}_{key}'
            flatten_dict(val, new_key, l)
    else:
        l.append((k, v))

    return dict(l)


def is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def merge_values(values: list[Any], key: str, require_all: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if all(isinstance(v, Mapping) for v in values):
        result[key] = _merge_dicts(values, require_all=require_all)

    elif all(is_integer(v) for v in values) and len(set(values)) == 1:
        result[key] = values[0]

    elif all(is_number(v) for v in values):
        nums = [float(v) for v in values]
        result[key] = mean(nums)
        result[f'{key}_std'] = stdev(nums) if len(nums) > 1 else 0.0

    elif all(isinstance(v, (str, bool)) for v in values):
        first = values[0]
        if all(v == first for v in values):
            result[key] = first

    return result


def _merge_dicts(dicts: list[Mapping[str, Any]], require_all: bool = False) -> dict[str, Any]:
    if not dicts:
        return {}

    if require_all:
        keys = set(dicts[0])
        for d in dicts[1:]:
            keys &= set(d)
    else:
        keys = set().union(*(d.keys() for d in dicts))

    result: dict[str, Any] = {}

    for key in keys:
        values = [d[key] for d in dicts if key in d]
        result.update(merge_values(values, key, require_all=require_all))

    return result


def merge_dicts(*dicts: Mapping[str, Any], require_all: bool = False) -> dict[str, Any]:
    return _merge_dicts(list(dicts), require_all=require_all)
