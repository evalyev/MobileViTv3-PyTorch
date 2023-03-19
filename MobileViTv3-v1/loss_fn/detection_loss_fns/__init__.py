#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import importlib
import os
import argparse

from utils import logger

from ..base_criteria import BaseCriteria

SUPPORTED_DETECTION_LOSS_FNS = []
DETECTION_LOSS_FN_REGISTRY = {}


def register_detection_loss_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_DETECTION_LOSS_FNS:
            raise ValueError(
                "Cannot register duplicate detection loss function ({})".format(name)
            )

        if not issubclass(cls, BaseCriteria):
            raise ValueError(
                "Loss function ({}: {}) must extend BaseCriteria".format(
                    name, cls.__name__
                )
            )

        DETECTION_LOSS_FN_REGISTRY[name] = cls
        SUPPORTED_DETECTION_LOSS_FNS.append(name)
        return cls

    return register_fn


def supported_loss_fn_str(loss_fn_name):
    supp_str = "Loss function ({}) is not yet supported. \n Supported functions for detection are:".format(
        loss_fn_name
    )
    for i, fn_name in enumerate(SUPPORTED_DETECTION_LOSS_FNS):
        supp_str += "{} \t".format(fn_name)
    logger.error(supp_str)


def arguments_detection_loss_fn(parser: argparse.ArgumentParser):
    # add loss function specific arguments
    for k, v in DETECTION_LOSS_FN_REGISTRY.items():
        parser = v.add_arguments(parser=parser)
    return parser


def get_detection_loss(opts):
    loss_fn_name = getattr(opts, "loss.detection.name", "cross_entropy")

    if loss_fn_name in SUPPORTED_DETECTION_LOSS_FNS:
        return DETECTION_LOSS_FN_REGISTRY[loss_fn_name](opts)
    else:
        supported_loss_fn_str(loss_fn_name)
        return None


# automatically import different loss functions
loss_fn_dir = os.path.dirname(__file__)
for file in os.listdir(loss_fn_dir):
    path = os.path.join(loss_fn_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("loss_fn.detection_loss_fns." + model_name)
