"""Runnable companion plugin for sft_report_v2.md.

This example demonstrates two independent ms-swift extension points:

1. ``KeySpanLossScale`` gives text enclosed by ``<key>...</key>`` a larger
   token weight while retaining the ordinary assistant-response objective.
2. ``WeightedMeanCrossEntropyLoss`` normalizes by the sum of token weights,
   so changing the weight scale does not silently change the effective
   learning rate of a batch.

Load it with ``--external_plugins``.  Importing this file only registers the
two names; it does not start training or modify model weights.
"""

import re

import torch

from swift.loss import BaseLoss, loss_map
from swift.loss_scale import LossScale, loss_scale_map


class KeySpanLossScale(LossScale):
    """Up-weight marked key spans inside assistant responses."""

    is_binary = False
    _key_span = re.compile(r'(<key>.*?</key>)', flags=re.IGNORECASE | re.DOTALL)

    def get_loss_scale(self, context, **kwargs):
        if not isinstance(context, str):
            return [context], [1.0]

        parts = [part for part in self._key_span.split(context) if part]
        if not parts:
            return [context], [1.0]

        weights = [3.0 if self._key_span.fullmatch(part) else 1.0 for part in parts]
        return parts, weights


class WeightedMeanCrossEntropyLoss(BaseLoss):
    """Causal-LM cross entropy divided by the sum of valid token weights."""

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func

        token_loss = per_token_loss_func(outputs, labels)
        shifted_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1)
        valid_mask = shifted_labels.ne(-100).to(device=token_loss.device)

        if loss_scale is None:
            weights = valid_mask.to(dtype=token_loss.dtype)
        else:
            weights = loss_scale.to(device=token_loss.device, dtype=token_loss.dtype)
            weights = weights * valid_mask.to(dtype=token_loss.dtype)

        denominator = weights.sum().clamp_min(1.0)
        return (token_loss * weights).sum() / denominator


loss_scale_map['key_span'] = KeySpanLossScale
loss_map['weighted_mean_ce'] = WeightedMeanCrossEntropyLoss
