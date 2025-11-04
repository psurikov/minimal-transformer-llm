from dataclasses import dataclass

@dataclass
class TrainingParameters:
    vocab_size: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    rope_theta: float
    norm_eps: float
    device_str: str
    min_learning_rate: float
    max_learning_rate: float
    iterations: int
    warmup_iterations: int
    cosine_iterations: int
    train_tokens_file: str
    valid_tokens_file: str
    load_checkpoint_file: str
    save_checkpoint_file: callable
