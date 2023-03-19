#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
from torch.nn import functional as F
from typing import Optional, Dict, List
import io
import os
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from contextlib import redirect_stdout

from cvnets.models.detection.base_detection import DetectionPredTuple
from utils.tensor_utils import all_gather_list
from utils import logger
from utils.ddp_utils import is_master

from . import register_stats_fn


@register_stats_fn(name="coco_map")
class COCOEvaluator(object):
    def __init__(
        self,
        opts,
        split: Optional[str] = "val",
        year: Optional[int] = 2017,
        use_distributed: Optional[bool] = False,
        *args,
        **kwargs
    ):
        # disable printing on console, so that pycocotools print statements are not printed on console
        logger.disable_printing()
        bkrnd_id = (
            0 if getattr(opts, "dataset.detection.no_background_id", False) else 1
        )

        iou_types = getattr(opts, "stats.coco_map.iou_types", ["bbox"])

        root = getattr(opts, "dataset.root_val", None)
        ann_file = os.path.join(
            root, "annotations/instances_{}{}.json".format(split, year)
        )
        coco_gt = COCO(ann_file)

        coco_categories = sorted(coco_gt.getCatIds())
        self.coco_id_to_contiguous_id = {
            coco_id: i + bkrnd_id for i, coco_id in enumerate(coco_categories)
        }
        self.contiguous_id_to_coco_id = {
            v: k for k, v in self.coco_id_to_contiguous_id.items()
        }

        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.use_distributed = use_distributed
        self.is_master_node = is_master(opts)

        self.coco_results = {iou_type: [] for iou_type in iou_types}

        # enable printing, to enable cvnets log printing
        logger.enable_printing()

    def prepare_predictions(self, predictions: Dict, targets: List):
        if not (
            isinstance(predictions, Dict)
            and ({"detections"} <= set(list(predictions.keys())))
        ):
            logger.error(
                "For coco evaluation during training, the output from the model should be a dictionary "
                "and should contain the results in a key called detections"
            )

        detections = predictions["detections"]

        if isinstance(targets, list):
            image_ids = torch.tensor(
                [t["image_id"] for t in targets], dtype=torch.int64
            )
            image_widths = torch.tensor(
                [t["image_width"] for t in targets], dtype=torch.int64
            )
            image_heights = torch.tensor(
                [t["image_height"] for t in targets], dtype=torch.int64
            )
        else:
            image_ids = targets["image_id"]
            image_widths = targets["image_width"]
            image_heights = targets["image_height"]

        if isinstance(detections, DetectionPredTuple):
            detections = [detections]

        if not (
            isinstance(detections, List)
            and isinstance(detections[0], DetectionPredTuple)
        ):
            logger.error(
                "For coco evaluation during training, the results should be stored as a List of DetectionPredTuple"
            )

        self.prepare_cache_results(
            detection_results=detections,
            image_ids=image_ids,
            image_widths=image_widths,
            image_heights=image_heights,
        )

    def prepare_cache_results(
        self,
        detection_results: List[DetectionPredTuple],
        image_ids,
        image_widths,
        image_heights,
    ) -> None:
        batch_results = {k: [] for k in self.coco_results.keys()}
        for detection_result, img_id, img_w, img_h in zip(
            detection_results, image_ids, image_widths, image_heights
        ):
            label = detection_result.labels

            if label.numel() == 0:
                # no detections
                continue
            box = detection_result.boxes
            score = detection_result.scores

            img_id, img_w, img_h = img_id.item(), img_w.item(), img_h.item()

            box[..., 0::2] = torch.clip(box[..., 0::2] * img_w, min=0, max=img_w)
            box[..., 1::2] = torch.clip(box[..., 1::2] * img_h, min=0, max=img_h)

            # convert box from xyxy to xywh format
            box[..., 2] = box[..., 2] - box[..., 0]
            box[..., 3] = box[..., 3] - box[..., 1]

            box = box.cpu().numpy()
            label = label.cpu().numpy()
            score = score.cpu().numpy()

            if "bbox" in batch_results:
                batch_results["bbox"].extend(
                    [
                        {
                            "image_id": img_id,
                            "category_id": self.contiguous_id_to_coco_id[
                                label[bbox_id]
                            ],
                            "bbox": box[bbox_id].tolist(),
                            "score": score[bbox_id],
                        }
                        for bbox_id in range(box.shape[0])
                        if label[bbox_id] > 0
                    ]
                )

            masks = detection_result.masks
            if masks is not None and "segm" in batch_results:
                # masks are [N, H, W]. For interpolation, convert them to [1, N, H, W] and then back to [N, H, W]
                masks = F.interpolate(
                    masks.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=True
                ).squeeze(0)
                masks = masks > 0.5

                masks = masks.cpu().numpy()
                # predicted masks are in [N, H, W] format
                rles = [
                    maskUtils.encode(
                        np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
                    )[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

                batch_results["segm"].extend(
                    [
                        {
                            "image_id": img_id,
                            "category_id": self.contiguous_id_to_coco_id[label[seg_id]],
                            "segmentation": rle,
                            "score": score[seg_id],
                        }
                        for seg_id, rle in enumerate(rles)
                        if label[seg_id] > 0
                    ]
                )

        for k in batch_results.keys():
            self.coco_results[k].extend(batch_results[k])

    def gather_coco_results(self) -> None:
        # synchronize results across different devices
        for iou_type, coco_results in self.coco_results.items():
            # agg_coco_results as List[List].
            # The outer list is for processes and inner list is for coco_results in the process
            if self.use_distributed:
                agg_coco_results = all_gather_list(coco_results)

                merged_coco_results = []
                # filter the duplicates
                for (
                    p_coco_results
                ) in agg_coco_results:  # retrieve results from each process
                    merged_coco_results.extend(p_coco_results)
            else:
                merged_coco_results = coco_results

            self.coco_results[iou_type] = merged_coco_results

    def summarize_coco_results(self) -> Dict:
        stats_map = dict()
        for iou_type, coco_results in self.coco_results.items():
            if len(coco_results) < 1:
                # during initial epochs, we may not have any sample results, so we can skip this part
                map_val = 0.0
            else:
                try:
                    logger.disable_printing()

                    with redirect_stdout(io.StringIO()):
                        coco_dt = COCO.loadRes(self.coco_gt, coco_results)

                    coco_eval = COCOeval(
                        cocoGt=self.coco_gt, cocoDt=coco_dt, iouType=iou_type
                    )
                    coco_eval.evaluate()
                    coco_eval.accumulate()

                    if self.is_master_node:
                        logger.enable_printing()

                    logger.log("Results for IoU Metric: {}".format(iou_type))
                    coco_eval.summarize()
                    map_val = coco_eval.stats[0].item()
                except Exception as e:
                    map_val = 0.0
            stats_map[iou_type] = map_val * 100

        logger.enable_printing()
        return stats_map
