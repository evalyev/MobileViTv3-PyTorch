#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from typing import Any
from utils import logger

from . import BaseCriteria, register_loss_fn
from .multi_modal_img_text_loss_fns import (
    get_multi_modal_img_text_loss,
    arguments_multi_modal_img_text_loss_fn,
)


@register_loss_fn("multi_modal_image_text")
class MultiModalImageTextLoss(BaseCriteria):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.criteria = get_multi_modal_img_text_loss(opts=opts, *args, **kwargs)

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Any:
        return self.criteria(
            input_sample=input_sample, prediction=prediction, target=target
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.multi-modal-image-text.name",
            type=str,
            default="clip",
            help="Loss function name",
        )
        parser = arguments_multi_modal_img_text_loss_fn(parser)
        return parser

    def __repr__(self):
        return self.criteria.__repr__()
