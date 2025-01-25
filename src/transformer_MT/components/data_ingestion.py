import os
import gdown
import zipfile
from pathlib import Path
from src.transformer_MT import logger
from src.transformer_MT.utils.common import get_size
from src.transformer_MT.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            gdown.download(self.config.source_URL, self.config.local_data_file, quiet=False)
            logger.info(f"{self.config.local_data_file} downloaded successfully")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)