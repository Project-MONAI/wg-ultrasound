import torch
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric,
)
from monai.utils import MetricReduction
from monai.networks.utils import one_hot
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
        
        The function requires a batch dimension or will auto-batch a (C, H, W) input
        if C matches the number of classes. It retains use of monai.networks.utils.one_hot
        for (B, 1, spatial...) inputs.
        """
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
            
        if tensor.dim() == 3:
            # Detect (C, H, W) vs (B, H, W)
            if tensor.shape[0] == num_classes and num_classes > 1:
                # Treat as unbatched multi-channel (C, H, W)
                tensor = tensor.unsqueeze(0)
            else:
                # Treat as batched single-channel/indices (B, H, W)
                tensor = tensor.unsqueeze(1)
             
        if tensor.shape[1] == 1 and num_classes > 1:
            # Assuming tensor contains class indices or binary mask
            tensor = one_hot(tensor.long(), num_classes=num_classes)
            
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
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
            
        # Normalize dims before discretizing
        if y_pred.dim() == 3:
            if y_pred.shape[0] == self.num_classes and self.num_classes > 1:
                y_pred = y_pred.unsqueeze(0)
            else:
                y_pred = y_pred.unsqueeze(1)
                
        if discretize:
            if self.num_classes == 2:
                # Binary: Simple threshold on (B, 1, H, W) probs
                y_pred = (y_pred >= 0.5).float()
            else:
                # Multi-class: Argmax over channel dimension and convert back to one-hot
                class_indices = torch.argmax(y_pred, dim=1, keepdim=True)
                y_pred = one_hot(class_indices, num_classes=self.num_classes)
        
        # Ensure final one-hot format for metrics
        y_pred = self._ensure_one_hot(y_pred, self.num_classes)
        y = self._ensure_one_hot(y, self.num_classes)

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
        import warnings
        
        def safe_item(metric_obj, name):
            val = metric_obj.aggregate()
            # Handle ConfusionMatrixMetric which returns a list
            if isinstance(val, (list, tuple)):
                results = []
                for v in val:
                    if torch.isfinite(v).all():
                        results.append(v.item())
                    else:
                        warnings.warn(f"Non-finite value detected in {name}", stacklevel=2)
                        fallback = float('inf') if name in ["HD95", "ASD"] else 0.0
                        results.append(fallback)
                return results
            
            if torch.isfinite(val).all():
                return val.item()
            else:
                warnings.warn(f"Non-finite value detected in {name}", stacklevel=2)
                return float('inf') if name in ["HD95", "ASD"] else 0.0

        metrics = {
            "Dice": safe_item(self.dice_metric, "Dice"),
            "IoU": safe_item(self.iou_metric, "IoU"),
            "HD95": safe_item(self.hd95_metric, "HD95"),
            "ASD": safe_item(self.asd_metric, "ASD"),
        }
        
        cm_results = safe_item(self.confusion_matrix, "ConfusionMatrix")
        if isinstance(cm_results, list) and len(cm_results) >= 2:
            metrics["Precision"] = cm_results[0]
            metrics["Recall"] = cm_results[1]
        
        return metrics

    def get_results_df(self):
        """Returns the computed metrics as a formatting Pandas DataFrame."""
        results = self.compute()
        return pd.DataFrame([results])
