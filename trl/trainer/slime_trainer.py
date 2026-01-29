import torch
import torch.nn.functional as F
from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Union

from transformers import PreTrainedModel

class SlimeTrainer(DPOTrainer):
    """
    Trainer class for Slime-based DPO training.

    Inherits from DPOTrainer and overrides loss and forward pass methods
    to implement Slime-specific logic.
    """

    def __init__(self, args: Any, *other_args: Any, **kwargs: Any) -> None:
        """
        Initialize the SlimeTrainer.

        Args:
            args (SlimeConfig): The configuration object containing
                training arguments.
            *other_args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(args=args, *other_args, **kwargs)

        if "ipo" not in self.loss_type:
            self.loss_type = "ipo"

    def concatenated_forward(
            self,
            model: Union[PreTrainedModel, torch.nn.Module],
            batch: Dict[str, Union[torch.Tensor, Any]],
            is_ref_model: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Run the forward pass on concatenated inputs.

        Args:
            model: The model to run the forward pass on.
            batch: The batch of data.
            is_ref_model: Whether this is the reference model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing log probabilities
            and logits metrics.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            padding_value=self.pad_token_id
        )

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        prompt_att_mask = concatenated_batch["prompt_attention_mask"]
        completion_att_mask = concatenated_batch["completion_attention_mask"]

        input_ids = torch.cat(
            (prompt_input_ids, completion_input_ids), dim=1
        )
        attention_mask = torch.cat(
            (prompt_att_mask, completion_att_mask), dim=1
        )

        loss_mask = torch.cat(
            (torch.zeros_like(prompt_att_mask), completion_att_mask),
            dim=1,
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        )
        logits = outputs.logits

        # --- Log Probability Calculation ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)

        per_token_logps = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        ).squeeze(2)

        per_token_logps = per_token_logps * shift_mask

        len_chosen = batch["chosen_input_ids"].shape[0]

        chosen_token_logps = per_token_logps[:len_chosen]
        rejected_token_logps = per_token_logps[len_chosen:]

        all_logps = per_token_logps.sum(-1)
        completion_lens = shift_mask.sum(-1)
        all_logps = all_logps / completion_lens

        logits_chosen = logits[:len_chosen]
        logits_rejected = logits[len_chosen:]
        mask_chosen = loss_mask[:len_chosen]
        mask_rejected = loss_mask[len_chosen:]

        mean_chosen_logits = logits_chosen[mask_chosen.bool()].mean()
        mean_rejected_logits = logits_rejected[mask_rejected.bool()].mean()

        return {
            "chosen_logps": all_logps[:len_chosen],
            "rejected_logps": all_logps[len_chosen:],
            "chosen_token_logps": chosen_token_logps,
            "rejected_token_logps": rejected_token_logps,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
        }

    def dpo_loss(
            self,
            chosen_logps: torch.FloatTensor,
            rejected_logps: torch.FloatTensor,
            ref_chosen_logps: torch.FloatTensor,
            ref_rejected_logps: torch.FloatTensor,
            loss_type: str = "sigmoid",
            model_output: Dict[str, Any] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Calculate the custom Slime DPO loss.

        Args:
            chosen_logps: Log probabilities of chosen responses.
            rejected_logps: Log probabilities of rejected responses.
            ref_chosen_logps: Reference model log probs (unused in Slime).
            ref_rejected_logps: Reference model log probs (unused in Slime).
            loss_type: The type of loss (default: sigmoid).
            model_output: Dictionary containing token-level logs.

        Returns:
            Tuple containing (loss, chosen_logps, rejected_logps).
        """
        # 1. Distance Term
        dist = chosen_logps - rejected_logps

        dist_loss_linear = F.relu(-dist + self.args.hard_margin)
        dist_loss_soft_der = torch.sigmoid(
            -2.5 * (dist - self.args.soft_margin)
        )
        dist_loss = self.args.dist_lambda * (
                dist_loss_linear * dist_loss_soft_der
        )

        # 2. Chosen Penalty
        chosen_penalty = (
                -self.args.center_lambda_chosen * chosen_logps
        )

        # 3. Rejected Penalty (Token-level)
        if model_output is None:
            raise ValueError(
                "model_output must be provided for Slime loss calculation."
            )

        rejected_token_logps = model_output["rejected_token_logps"]
        shift = self.args.rejected_penalty_shift

        mask = (rejected_token_logps != 0).float()

        inner = -rejected_token_logps - shift
        penalty_matrix = (F.softplus(inner)) ** 2.5

        rejected_penalty = self.args.center_lambda_rejected * (
                (penalty_matrix * mask).sum() / (mask.sum() + 1e-8)
        )

        total_loss = (dist_loss + chosen_penalty + rejected_penalty).mean()

        return (
            total_loss,
            chosen_logps.detach(),
            rejected_logps.detach()
        )
