import torch
import torch.optim as optim
import pickle
from torch import nn
from pathlib import Path
from src.transformer_MT.config.configuration import ConfigurationManager
from src.transformer_MT.components.data_loader import MTDataLoader
from src.transformer_MT.components.model.transformer import Transformer
from src.transformer_MT.components.training import Training
from src.transformer_MT.constants import PAD_IDX

class TrainingPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def main(self):
        config = ConfigurationManager()
        token_vocab_transform_config = config.get_token_vocab_transform_config()
        data_loader_config = config.get_mt_data_loader_config()
        data_loader = MTDataLoader(config=data_loader_config)
        data_loader.load_transforms(token_vocab_transform_config.root_dir)

        vocab_transform = data_loader.get_vocab_transform()

        src_vocab_size = len(vocab_transform["en"])
        tgt_vocab_size = len(vocab_transform["vi"])

        training_config = config.get_training_config()

        transformer = Transformer(src_vocab_size, tgt_vocab_size, training_config.d_model, 
                                  training_config.num_heads, training_config.num_layers, training_config.d_ff, 
                                  training_config.max_seq_length, training_config.dropout, self.device).to(self.device)
        
        optimizer = optim.Adam(transformer.parameters(), lr=training_config.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        save_dir = Path(data_loader_config.root_dir)

        with open(save_dir / 'train_loader.pkl', 'rb') as f:
            train_loader = pickle.load(f)

        with open(save_dir / 'val_loader.pkl', 'rb') as f:
            val_loader = pickle.load(f)

        training = Training(training_config, transformer, train_loader, val_loader, optimizer, criterion, self.device, tgt_vocab_size)
        
        training.train()