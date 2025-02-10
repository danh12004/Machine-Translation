import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from src.transformer_MT.config.configuration import ConfigurationManager
from src.transformer_MT.components.data_loader import MTDataLoader
from src.transformer_MT.utils.common import load_json
from src.transformer_MT import logger

class DataLoaderTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        token_vocab_transform_config = config.get_token_vocab_transform_config()
        data_loader_config = config.get_mt_data_loader_config()
        
        data_loader = MTDataLoader(config=data_loader_config)
        
        data_loader.load_transforms(token_vocab_transform_config.root_dir)
        
        data_loader.create_text_transform()

        loaded_train = load_json(Path(data_preprocessing_config.root_dir) / 'train.json')
        loaded_val = load_json(Path(data_preprocessing_config.root_dir) / 'val.json')
        loaded_test = load_json(Path(data_preprocessing_config.root_dir) / 'test.json')

        train_loader = DataLoader(loaded_train, batch_size=data_loader_config.batch_size, shuffle=True, collate_fn=data_loader.collate_fn)
        val_loader = DataLoader(loaded_val, batch_size=data_loader_config.batch_size, shuffle=False, collate_fn=data_loader.collate_fn)
        test_loader = DataLoader(loaded_test, batch_size=data_loader_config.batch_size, shuffle=False, collate_fn=data_loader.collate_fn)

        save_dir = Path(data_loader_config.root_dir)

        with open(save_dir / 'train_loader.pkl', 'wb') as f:
            pickle.dump(train_loader, f)

        with open(save_dir / 'val_loader.pkl', 'wb') as f:
            pickle.dump(val_loader, f)

        with open(save_dir / 'test_loader.pkl', 'wb') as f:
            pickle.dump(test_loader, f)

        logger.info("Save train, val, test loaders successfully!")
