from dataclasses import dataclass    
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_files: dict

@dataclass(frozen=True)
class TokenVocabTransformConfig:
    root_dir: Path

@dataclass(frozen=True)
class MTDataLoaderConfig:
    root_dir: Path
    batch_size: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    max_seq_length: int
    dropout: int
    learning_rate: int 
    num_epochs: int
    clip: int