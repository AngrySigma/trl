import torch
import torch.nn.functional as F
from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

from transformers import PreTrainedModel


@dataclass
class SlimeConfig(DPOConfig):
    rejected_penalty_shift: float = field(
        default=0.5,
        metadata={"help": "The shift applied to the penalty for rejected responses."}
    )
    center_lambda_rejected: float = field(
        default=0.0005,
        metadata={"help": "Regularization weight for the center of rejected responses."}
    )
    center_lambda_chosen: float = field(
        default=0.8,
        metadata={"help": "Regularization weight for the center of chosen responses."}
    )
    soft_margin: float = field(
        default=1.0,
        metadata={"help": "Soft margin value for the loss function."}
    )
    hard_margin: float = field(
        default=1.5,
        metadata={"help": "Hard margin value for the loss function."}
    )
    dist_lambda: float = field(
        default=1.0,
        metadata={"help": "Weighting factor for the distance-based penalty."}
    )

    def __post_init__(self):
        super().__post_init__()