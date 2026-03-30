from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class RuGPT3XLConfig(PretrainedConfig):
    """Configuration class for the RuGPT-3 XL model (1.3B parameters).

    This is a GPT-3-style decoder-only transformer trained on Russian text by
    SberDevices. Architecture: learned absolute position embeddings, pre-norm
    transformer layers with GELU activation, and tied word embeddings for the
    language modeling head.
    """

    model_type = "rugpt3xl"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50264,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        hidden_act="gelu_new",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        output_dropout=0.1,
        use_cache=True,
        bos_token_id=2,
        eos_token_id=1,
        pad_token_id=0,
        tie_word_embeddings=False,
        sparse_mode="none",
        sparse_block_size=16,
        sparse_num_local_blocks=8,
        sparse_num_global_blocks=1,
        sparse_num_different_global_patterns=8,
        attn_implementation="sdpa",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.use_cache = use_cache
        self.sparse_mode = sparse_mode
        self.sparse_block_size = sparse_block_size
        self.sparse_num_local_blocks = sparse_num_local_blocks
        self.sparse_num_global_blocks = sparse_num_global_blocks
        self.sparse_num_different_global_patterns = sparse_num_different_global_patterns
        self.attn_implementation = attn_implementation

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
