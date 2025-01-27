import os
import re
import html
from pathlib import Path
from src.transformer_MT import logger
from src.transformer_MT.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def load_data(self, en_file_path: Path, vi_file_path: Path):
        try:
            if not os.path.exists(en_file_path) or not os.path.exists(vi_file_path):
                raise FileNotFoundError(f"File not found: {en_file_path}, {vi_file_path}")
            
            with open(en_file_path, 'r', encoding='utf-8') as en_file, open(vi_file_path, 'r', encoding='utf-8') as vi_file:
                english_lines = [line.strip() for line in en_file.readlines()]
                vietnamese_lines = [line.strip() for line in vi_file.readlines()]

            if len(english_lines) != len(vietnamese_lines):
                raise ValueError(f"The number of lines in files do not match.")
            
            logger.info(f"Loaded {len(english_lines)} lines.")
            return [(en, vi) for en, vi in zip(english_lines, vietnamese_lines)]
        except KeyError:
            raise ValueError(f"Must be one of {list(self.config.data_files.keys())}.")
        
    def clean_data(self, text):
        text = text.lower()

        text = html.unescape(text)
        text = re.sub(r'[^\w\s\u00C0-\u1FFF\u2C00-\uD7FF]', '', text)
        text = re.sub(r'\s+', ' ', text.strip()) 

        return text

        