from dataclasses import dataclass

@dataclass
class LMconfig():
        dim: int = 512  # 模型维度，默认为 512
        n_layers: int = 20  # 层数，默认为 8
        head_num: int = 16  # 默认为 16
        vocab_size: int = 6400 # 词汇表大小，默认为 6400
        eps: float = 1e-5 # 归一化层的 epsilon 值，默认为 1e-5
        max_seq_len: int = 512#  512
        dropout: float = 0.1 # 默认为 0.0
        if_falsh:bool = True
        use_moe = False
