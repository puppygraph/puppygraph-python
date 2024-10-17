"""Utility functions for working with dataclasses."""

from dataclasses import asdict
from enum import Enum


def dataclass_to_camel_dict(instance) -> dict:
    """Convert a dataclass instance to a camelCase dictionary, omitting fields with None values."""
    if not hasattr(instance, "__dataclass_fields__"):
        raise ValueError("The provided instance is not a dataclass.")

    def _snake_to_camel(snake_str: str) -> str:
        """Convert a snake_case string to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    def _convert_value(value):
        """Convert Enum to its name, and keep other types as they are."""
        if isinstance(value, Enum):
            return value.name
        return value

    return {
        _snake_to_camel(k): _convert_value(v)
        for k, v in asdict(instance).items()
        if v is not None
    }
