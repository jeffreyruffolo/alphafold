from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

import importlib_metadata
import numpy as np
import tensorflow as tf

from alphafold.model.features import FeatureDict
from alphafold.model import model
from alphafold.model.tf import shape_placeholders

os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "2.0"

NUM_RES = shape_placeholders.NUM_RES
NUM_MSA_SEQ = shape_placeholders.NUM_MSA_SEQ
NUM_EXTRA_SEQ = shape_placeholders.NUM_EXTRA_SEQ
NUM_TEMPLATES = shape_placeholders.NUM_TEMPLATES


def make_fixed_size(
    protein: Mapping[str, Any],
    shape_schema,
    msa_cluster_size: int,
    extra_msa_size: int,
    num_res: int,
    num_templates: int = 0,
) -> FeatureDict:
    """Guess at the MSA and sequence dimensions to make fixed size."""

    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
        NUM_EXTRA_SEQ: extra_msa_size,
        NUM_TEMPLATES: num_templates,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)

        schema = shape_schema[k]

        assert len(shape) == len(schema), (
            f"Rank mismatch between shape and shape schema for {k}: "
            f"{shape} vs {schema}")
        pad_size = [
            pad_size_map.get(s2, None) or s1
            for (s1, s2) in zip(shape, schema)
        ]
        padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]

        if padding:
            # TODO: alphafold's typing is wrong
            protein[k] = tf.pad(v, padding, name=f"pad_to_fixed_{k}")
            protein[k].set_shape(pad_size)
    return {k: np.asarray(v) for k, v in protein.items()}


def batch_input(
    input_features: model.features.FeatureDict,
    model_runner: model.RunModel,
    model_name: str,
    crop_len: int,
    use_templates: bool,
) -> model.features.FeatureDict:
    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    # templates models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = eval_cfg.max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = eval_cfg.max_msa_clusters

    max_msa_clusters = pad_msa_clusters

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
        extra_msa_size=5120,  # extra_msa (4, 5120, 68)
        num_res=crop_len,  # aatype (4, 68)
        num_templates=4,
    )  # template_mask (4, 4) second value
    return input_fix