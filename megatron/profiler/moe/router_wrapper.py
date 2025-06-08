# Copyright (c) 2024, Yicheng Feng. All rights reserved.

from typing import Callable, Optional, Tuple, List

import torch
from functools import partial

from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

# --- Main Wrapper Class ---

class TopKRouterWrapper(TopKRouter):
    """
    A wrapper for the TopKRouter that allows for custom routing functions.

    This class extends the standard TopKRouter by providing an API to inject a
    custom load balancing algorithm. If a custom function is provided, it will
    be used for routing decisions; otherwise, it defaults to the parent class's
    routing logic.
    """

    def __init__(
        self,
        config: TransformerConfig,
        custom_routing_func: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> None:
        """
        Initialize the TopKRouterWrapper.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            custom_routing_func (Optional[Callable]): A custom function that takes logits
                as input and returns a tuple of (scores, indices). Defaults to None.
        """
        super().__init__(config=config)
        self.custom_routing_func = custom_routing_func

    def routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overrides the default routing logic to use a custom routing function if provided.

        Args:
            logits (torch.Tensor): Logits tensor from the gating network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the scores and indices
                for expert selection.
        """
        if self.custom_routing_func is not None:
            # If a custom function is provided, call it directly.
            # It completely replaces the parent's routing logic.
            return self.custom_routing_func(logits)
        else:
            # If no custom function is provided, fall back to the original routing behavior.
            return super().routing(logits)


