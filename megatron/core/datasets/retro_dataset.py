# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import (
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import MMapIndexedDataset
from megatron.core.datasets.utils import Split, log_single_rank

logger = logging.getLogger(__name__)


# >>>
# @dataclass(kw_only=True)
@dataclass
# <<<
class RetroDatasetConfig(GPTDatasetConfig):
    """Configuration object for Megatron Core blended and megatron Retro datasets

    Attributes:
        return_document_ids (bool): Whether to return the document ids when querying the dataset.
        Turn this option on during preprocessing.

        split_preprocessing (str): The Retro preprocessing split string. It follows the same
        pattern convention as 'split'. Not to be used with 'blend_per_split'.
    """

    return_document_ids: bool = None

    split_preprocessing: str = None

    def __post_init__(self):
        super().__post_init__()
        assert self.split is not None, "the Retro data pipeline does not support 'blend_per_split'"
        split_vector = parse_and_normalize_split(self.split)
        split_preprocessing_vector = parse_and_normalize_split(self.split_preprocessing)
        if not numpy.allclose(split_vector, split_preprocessing_vector):
            self.split_matrix = convert_split_vector_to_split_matrix(
                split_vector, split_preprocessing_vector
            )
            log_single_rank(
                logger,
                logging.WARNING,
                f"split =/= split_preprocessing. Let split_matrix = {self.split_matrix}",
            )


class RetroDataset(GPTDataset):
    """The base Retro dataset

    Args:
        indexed_dataset (MMapIndexedDataset): The MMapIndexedDataset around which to build the
        MegatronDataset

        indexed_indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (int): The number of samples to draw from the indexed dataset

        index_split (Split): The indexed_indices Split

        config (RetroDatasetConfig): The Retro-specific container for all config sourced parameters
    """

    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: numpy.ndarray,
        num_samples: int,
        index_split: Split,
        config: RetroDatasetConfig,
    ) -> None:
        super().__init__(indexed_dataset, indexed_indices, num_samples, index_split, config)

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        """Abstract method implementation

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, numpy.ndarray]: The text ids and (optionally) the document ids wrapped in a
            dictionary
        """
        # >>>
        # from megatron import get_args
        # args = get_args()
        # if args.retro_fix_sub_epoch:
        #     idx = idx % len(self)
        # <<<
        text, document_ids = self._query_document_sample_shuffle_indices(idx)
        if getattr(self.config, "return_document_ids"):
            return {"text": text, "document_ids": document_ids}
        else:
            return {"text": text}

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Inherited method implementation

        The preprocessing split used for preprocessing will constrain the samples available for 
        pretraining.

        Returns:
            List[str]: The key config attributes
        """
        return super(RetroDataset, RetroDataset)._key_config_attributes() + ["split_preprocessing"]
