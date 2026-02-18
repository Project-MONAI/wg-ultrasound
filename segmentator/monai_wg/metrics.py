import torch
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.utils import MetricReduction
import pandas as pd
import numpy as np

class MonaiMetricWrapper:
    def __init__(self, num_classes=2, include_background=False, percentile=95):
        """
        Wrapper for various MONAI segmentation metrics.
        
        Args:
            num_classes (int): Number of classes in the segmentation.
            include_background (bool): Whether to include the background class in the metrics.
            percentile (int): Percentile for Hausdorff Distance (default 95).
        """
        self.num_classes = num_classes
        self.include_background = include_background
        self.percentile = percentile
        
        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=include_background, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.iou_metric = MeanIoU(include_background=include_background, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.hd95_metric = HausdorffDistanceMetric(include_background=include_background, percentile=percentile, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.asd_metric = SurfaceDistanceMetric(include_background=include_background, reduction=MetricReduction.MEAN, get_not_nans=False)
        self.confusion_matrix = ConfusionMatrixMetric(include_background=include_background, metric_name=["precision", "recall"], reduction=MetricReduction.MEAN, get_not_nans=False)

    def reset(self):
        """Reset all internal metric states."""
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd95_metric.reset()
        self.asd_metric.reset()
        self.confusion_matrix.reset()

    def _ensure_one_hot(self, tensor, num_classes):
        """
        Ensure the input tensor is in one-hot format (B, C, spatial...).
        If input is (B, 1, spatial...) or (B, spatial...), it converts to one-hot.
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
            
        if tensor.dim() == 3: # (B, H, W) Assuming 2D without channel dim
             tensor = tensor.unsqueeze(1)
             
        if tensor.shape[1] == 1 and num_classes > 1:
            # Assuming tensor contains class indices or binary mask
            from monai.networks.utils import one_hot
            tensor = one_hot(tensor, num_classes=num_classes)
            
        return tensor

    def update(self, y_pred, y, discretize=True):
        """
        Update metrics with new batch of data.
        
        Args:
            y_pred (torch.Tensor or np.ndarray): Prediction (Logits, Probabilities, or Class Indices).
                                                 Shape: (B, C, H, W) or (B, 1, H, W) or (B, H, W).
            y (torch.Tensor or np.ndarray): Ground Truth. shape similar to y_pred.
            discretize (bool): If True, discretizes y_pred (argmax/threshold) before computing metrics.
                               MONAI metrics usually handle this if setup, but ensuring consistency here.
        """
        y_pred = self._ensure_one_hot(y_pred, self.num_classes)
        y = self._ensure_one_hot(y, self.num_classes)
        
        # Discretize if needed (assuming probabilities/logits)
        if discretize:
            y_pred = (y_pred >= 0.5).float() # Simple threshold for handling mult-channel one-hot/binary

        # Update metrics
        self.dice_metric(y_pred=y_pred, y=y)
        self.iou_metric(y_pred=y_pred, y=y)
        self.hd95_metric(y_pred=y_pred, y=y)
        self.asd_metric(y_pred=y_pred, y=y)
        self.confusion_matrix(y_pred=y_pred, y=y)

    def compute(self):
        """
        Compute the final metric values.
        
        Returns:
            dict: Dictionary of metrics.
        """
        metrics = {
            "Dice": self.dice_metric.aggregate().item(),
            "IoU": self.iou_metric.aggregate().item(),
            "HD95": self.hd95_metric.aggregate().item(),
            "ASD": self.asd_metric.aggregate().item(),
        }
        
        # Confusion matrix returns list of [precision, recall]
        cm_results = self.confusion_matrix.aggregate()
        # cm_results is usually a list of tensors if multiple metrics requested
        # For precision and recall:
        metrics["Precision"] = cm_results[0].item()
        metrics["Recall"] = cm_results[1].item()
        
        return metrics

    def get_results_df(self):
        """Returns the computed metrics as a formatting Pandas DataFrame."""
        results = self.compute()
        return pd.DataFrame([results])
