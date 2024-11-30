from typing import Optional
from transformers import PretrainedConfig


class ReCentConfig(PretrainedConfig):
    model_type = "recent"

    def __init__(self,
                 model_name: str = "microsoft/deberta-v3-small",
                 max_width: int = 12,
                 hidden_size: int = 768,
                 dropout: float = 0.4,
                 vocab_size: int = -1,
                 max_neg_type_ratio: int = 2,
                 max_types: int = 15,
                 max_len: int = 100,
                 words_splitter_type: str = "whitespace",
                 class_token_index: int = -1,
                 encoder_config: Optional[dict] = None,
                 relation_token="[R]",
                 sep_token="[SEP]",
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (encoder_config["model_type"]
                                            if "model_type" in encoder_config
                                            else "deberta-v2")
            encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        self.encoder_config = encoder_config
        self.model_name = model_name
        self.max_width = max_width
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_neg_type_ratio = max_neg_type_ratio
        self.max_types = max_types
        self.max_len = max_len
        self.words_splitter_type = words_splitter_type
        self.class_token_index = class_token_index
        self.relation_token = relation_token
        self.sep_token = sep_token


# Register the configuration
from transformers import CONFIG_MAPPING

CONFIG_MAPPING.update({"recent": ReCentConfig})
