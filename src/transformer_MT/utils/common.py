import os
import yaml
import json

from box.exceptions import BoxValueError
from src.transformer_MT import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is not empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"creating directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: list):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info(f"json file save at: {path}")

@ensure_annotations
def load_json(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"json file loaded successfully from: {path}")
    return [(tuple(item)) for item in data]

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path))
    return f"~ {size_in_kb} KB"