import os
from src.transformer_MT.constants import *
from src.transformer_MT.utils.common import read_yaml, create_directories
from src.transformer_MT.entity import (DataIngestionConfig, DataPreprocessingConfig, TokenVocabTransformConfig, 
                                       MTDataLoaderConfig, TrainingConfig)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_files=config.data_files
        )

        return data_preprocessing_config
    
    def get_token_vocab_transform_config(self) -> TokenVocabTransformConfig:
        config = self.config.token_vocab_transform
        create_directories([config.root_dir])

        token_vocab_transformation_config = TokenVocabTransformConfig(root_dir=config.root_dir)

        return token_vocab_transformation_config
    
    def get_mt_data_loader_config(self) -> MTDataLoaderConfig:
        config = self.config.data_loader
        create_directories([config.root_dir])

        data_loader_config = MTDataLoaderConfig(root_dir=config.root_dir, batch_size=self.params.BATCH_SIZE)

        return data_loader_config
    
    def get_training_config(self):
        config = self.config.training
        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=config.root_dir,
            d_model=self.params.D_MODEL,
            num_heads=self.params.NUM_HEADS,
            num_layers=self.params.NUM_LAYERS,
            d_ff=self.params.D_FF,
            max_seq_length=self.params.MAX_SEQ_LENGTH,
            dropout=self.params.DROPOUT,
            learning_rate=self.params.LEARNING_RATE,
            num_epochs=self.params.NUM_EPOCHS,
            clip=self.params.CLIP
        )

        return training_config
