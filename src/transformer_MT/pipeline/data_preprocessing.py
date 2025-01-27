import os
from pathlib import Path
from src.transformer_MT.config.configuration import ConfigurationManager
from src.transformer_MT.components.data_preprocessing import DataPreprocessing
from src.transformer_MT.utils.common import save_json
from src.transformer_MT import logger

class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)

        train = data_preprocessing.load_data(
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.train.en, 
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.train.vi
        )

        val = data_preprocessing.load_data(
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.val.en, 
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.val.vi
        )

        test = data_preprocessing.load_data(
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.test.en, 
            Path(data_ingestion_config.unzip_dir) / data_preprocessing_config.data_files.test.vi
        )

        train = [(data_preprocessing.clean_data(source), data_preprocessing.clean_data(target)) for source, target in train]
        val = [(data_preprocessing.clean_data(source), data_preprocessing.clean_data(target)) for source, target in val]
        test = [(data_preprocessing.clean_data(source), data_preprocessing.clean_data(target)) for source, target in test]

        save_json(Path(data_preprocessing_config.root_dir) / 'train.json', train)
        save_json(Path(data_preprocessing_config.root_dir) / 'val.json', val)
        save_json(Path(data_preprocessing_config.root_dir) / 'test.json', test)

        # loaded_train = load_json(Path(data_preprocessing_config.root_dir) / 'train.json')
        # loaded_val = load_json(Path(data_preprocessing_config.root_dir) / 'val.json')
        # loaded_test = load_json(Path(data_preprocessing_config.root_dir) / 'test.json')

        