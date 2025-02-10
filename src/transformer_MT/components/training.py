import torch
import math, time
from src.transformer_MT.entity import TrainingConfig
from src.transformer_MT import logger

class Training:
    def __init__(self, config: TrainingConfig, model, train_loader, val_loader, optimizer, criterion, device, tgt_vocab_size):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tgt_vocab_size = tgt_vocab_size

    def train_one_epoch(self, clip):
        self.model.train()
        epoch_loss = 0

        for src, _, tgt in self.train_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, tgt[:, :-1])
            loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), tgt[:, 1:].contiguous().view(-1).long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for src, src_len, tgt in self.val_loader:
                src, src_len, tgt = src.to(self.device), src_len.to(self.device), tgt.to(self.device)
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(output.contiguous().view(-1, self.tgt_vocab_size), tgt[:, 1:].contiguous().view(-1).long())
                epoch_loss += loss.item()

        return epoch_loss / len(self.val_loader)
    
    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger(f"Model saved to {path}")

    def train(self):
        best_valid_loss = float('inf')
        clip = self.config.clip

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(clip)
            valid_loss = self.evaluate()

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_model(self.config.root_dir / 'model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
