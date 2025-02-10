from src.transformer_MT.constants import UNK_IDX, SPECIAL_SYMBOLS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.transformer_MT.entity import TokenVocabTransformConfig

class TokenVocabTransform():
    def __init__(self, config: TokenVocabTransformConfig):
        self.config = config
        self.languages = ["en", "vi"]
        self.token_transform = {}
        self.vocab_transform = {}

    def load_tokenizers(self):
        self.token_transform['en'] = get_tokenizer("spacy", language="en_core_web_sm")
        self.token_transform['vi'] = get_tokenizer("spacy", language="xx_ent_wiki_sm")

        return self.token_transform

    def yield_token(self, data, language):
        tokenizer = self.token_transform[language]
        for source, target in data:
            yield tokenizer(source if language == "en" else target)

    def build_vocab(self, data):
        for language in self.languages:
            self.vocab_transform[language] = build_vocab_from_iterator(
                self.yield_token(data, language),
                min_freq=2,
                specials=SPECIAL_SYMBOLS,
                special_first=True
            )

        for language in self.languages:
            self.vocab_transform[language].set_default_index(UNK_IDX)

        return self.vocab_transform