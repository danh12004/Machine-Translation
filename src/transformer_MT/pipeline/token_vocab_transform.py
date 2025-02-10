import pickle
from pathlib import Path
from src.transformer_MT.config.configuration import ConfigurationManager
from src.transformer_MT.components.token_vocab_transform import TokenVocabTransform
from src.transformer_MT.utils.common import load_json
from src.transformer_MT import logger

class TokenVocabTransformTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        loaded_train = load_json(Path(data_preprocessing_config.root_dir) / 'train.json')
        token_vocab_transform_config = config.get_token_vocab_transform_config()
        token_vocab_transform = TokenVocabTransform(config=token_vocab_transform_config)

        token_transform = token_vocab_transform.load_tokenizers()
        vocab_transform = token_vocab_transform.build_vocab(loaded_train)

        save_dir = Path(token_vocab_transform_config.root_dir)

        with open(save_dir / 'token_transform.pkl', 'wb') as f:
            pickle.dump(token_transform, f)

        with open(save_dir / 'vocab_transform.pkl', 'wb') as f:
            pickle.dump(vocab_transform, f)

        logger.info("Save token and vocab transform successfully!")













