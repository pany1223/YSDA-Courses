import typing as tp
import json
from decimal import Decimal


def call_cast(key_type: str) -> tp.Any:
    if key_type == 'int':
        return int
    elif key_type == 'float':
        return float
    elif key_type == 'decimal':
        return Decimal
    else:
        raise ValueError


def convert_recursively(deserialized: tp.Any) -> tp.Any:
    if isinstance(deserialized, dict):
        if '__custom_key_type__' in deserialized.keys():
            convert_to = deserialized['__custom_key_type__']
            deserialized = {call_cast(convert_to)(k): v
                            for k, v in deserialized.items()
                            if k != '__custom_key_type__'}
        return {k: convert_recursively(v) if isinstance(deserialized, dict) else v
                for k, v in deserialized.items()}
    elif isinstance(deserialized, list):
        return [convert_recursively(elem) for elem in deserialized]
    else:
        return deserialized


def decode_typed_json(json_value: str) -> tp.Any:
    """
    Returns deserialized object from json string.
    Checks __custom_key_type__ in object's keys to choose appropriate type.

    :param json_value: serialized object in json format
    :return: deserialized object
    """
    deserialized = json.loads(json_value)
    return convert_recursively(deserialized)
