from monai.transforms import (
    Compose,
    Resize,
    EnsureType,
    Activations,
    AsDiscrete,
    KeepLargestConnectedComponent,
)
import torch

class PostProcessingPipeline:
    def __init__(
        self,
        target_spacing=(1.0, 1.0, 1.0),
        target_spatial_size=(256, 256),
        activation=None,
        discretization="threshold",
        discretization_threshold=0.5,
        cleanup=False
    ):
        """
        Standardized Postprocessing Pipeline.
        
        Args:
            target_spacing (tuple): Target voxel spacing (mm). NOT USED currently in this simplified pipeline.
            target_spatial_size (tuple): Target spatial dimensions (H, W).
            activation (str, optional): 'sigmoid', 'softmax', or None.
            discretization (str, optional): 'threshold', 'argmax', or None.
            discretization_threshold (float): Threshold value if discretization is 'threshold'.
            cleanup (bool): Whether to apply KeepLargestConnectedComponent.
        """
        transforms_list = [EnsureType()]
        
        # 0. Resizing (Optional but usually needed for standardized metrics)
        if target_spatial_size:
            transforms_list.append(Resize(spatial_size=target_spatial_size, mode="nearest"))

        # 1. Activation
        if activation == "sigmoid":
            transforms_list.append(Activations(sigmoid=True))
        elif activation == "softmax":
            transforms_list.append(Activations(softmax=True))
        
        # 2. Discretization
        if discretization == "threshold":
            transforms_list.append(AsDiscrete(threshold=discretization_threshold))
        elif discretization == "argmax":
            transforms_list.append(AsDiscrete(argmax=True))
            
        # 3. Morphological Cleanup (Post-discretization)
        if cleanup:
            # applied_labels=1 assumes binary segmentation of interest is label 1
            transforms_list.append(KeepLargestConnectedComponent(applied_labels=[1]))
            
        self.transforms = Compose(transforms_list)
        
    def __call__(self, pred):
        return self.transforms(pred)

def get_standard_postprocessing(target_size=(256, 256)):
    """
    Simple helper to get the standard postprocessing transform.
    Defaults to Sigmoid -> Threshold(0.5) -> Cleanup.
    """
    return PostProcessingPipeline(
        target_spatial_size=target_size,
        activation="sigmoid",
        discretization="threshold",
        discretization_threshold=0.5,
        cleanup=True
    )
