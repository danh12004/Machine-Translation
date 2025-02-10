from pathlib import Path
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from src.transformer_MT.entity import MTDataLoaderConfig
from src.transformer_MT.constants import SOS_IDX, EOS_IDX, PAD_IDX
from src.transformer_MT import logger

class MTDataLoader:
    def __init__(self, config: MTDataLoaderConfig):
        self.config = config
        self.languages = ["en", "vi"]
        self.text_transform = {}  
        self.token_transform = {}
        self.vocab_transform = {}

    def get_vocab_transform(self):
        return self.vocab_transform

    def load_transforms(self, path):
        save_dir = Path(path)

        try:
            with open(save_dir / 'token_transform.pkl', 'rb') as f:
                self.token_transform = pickle.load(f)

            with open(save_dir / 'vocab_transform.pkl', 'rb') as f:
                self.vocab_transform = pickle.load(f)

            logger.info("Loaded token_transform and vocab_transform successfully!")
        except FileNotFoundError:
            logger.error("Saved token_transform or vocab_transform not found! You may need to train first.")

    def sequential_transform(self, text, transforms):
        for transform in transforms:
            text = transform(text)
        return text

    def tensor_transform(self, token_ids):
        return torch.cat([torch.tensor([SOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])])
    
    def apply_text_transform(self, text, language):
        transforms = [self.token_transform[language], self.vocab_transform[language], self.tensor_transform]
        return self.sequential_transform(text, transforms)
    
    def text_transform_func(self, text, language):
        return self.apply_text_transform(text, language)

    def create_text_transform(self):
        for language in self.languages:
            self.text_transform[language] = self.text_transform_func
            
    def collate_fn(self, batch):
        source_batch, source_length_batch, target_batch = [], [], []
        for source_sample, target_sample in batch:
            processed_source = self.text_transform["en"](source_sample)
            processed_target = self.text_transform["vi"](target_sample)
            source_batch.append(processed_source)
            target_batch.append(processed_target)
            source_length_batch.append(processed_source.size(0))

        source_batch = pad_sequence(source_batch, batch_first=True, padding_value=PAD_IDX)
        target_batch = pad_sequence(target_batch, batch_first=True, padding_value=PAD_IDX)
        return source_batch, torch.tensor(source_length_batch, dtype=torch.int64), target_batch
