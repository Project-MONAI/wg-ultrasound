import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns

def plot_segmentation(image, label, pred, alpha=0.5, save_path=None, title="Segmentation Overlay"):
    """
    Overlays Ground Truth and Prediction on the original Image.
    Supports Multi-class (distinct colors) and Binary segmentation.

    Args:
        image (np.ndarray | torch.Tensor): The background image. 
            Expected shapes: (H, W), (1, H, W), (3, H, W), or (H, W, 3).
        label (np.ndarray | torch.Tensor): Ground truth mask. 
            Expected shapes: (H, W), (1, H, W), or (C, H, W).
        pred (np.ndarray | torch.Tensor): Predicted mask. 
            Expected shapes: (H, W), (1, H, W), or (C, H, W).
        alpha (float, optional): Transparency of the overlay. Defaults to 0.5.
        save_path (str, optional): Path to save the figure if provided.
        title (str, optional): Title of the plot. Defaults to "Segmentation Overlay".
    """
    # Convert to numpy
    if isinstance(image, torch.Tensor): image = image.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()

    # Handle various image shapes (C, H, W) or (H, W, C)
    if image.ndim == 3:
        if image.shape[0] in [1, 3]: # (1, H, W) or (3, H, W)
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1: # (H, W, 1)
            image = np.squeeze(image, axis=-1)

    # Handle masks: (C, H, W) -> (H, W)
    if label.ndim == 3:
        if label.shape[0] == 1: # (1, H, W)
            label = np.squeeze(label, axis=0)
        else: # One-hot (C, H, W)
            label = np.argmax(label, axis=0)

    if pred.ndim == 3:
        if pred.shape[0] == 1: # (1, H, W)
            pred = np.squeeze(pred, axis=0)
        else: # One-hot (C, H, W)
            pred = np.argmax(pred, axis=0)

    label = (np.squeeze(label) > 0.5).astype(np.uint8) if label.max() <= 1 else np.squeeze(label)
    pred = (np.squeeze(pred) > 0.5).astype(np.uint8) if pred.max() <= 1 else np.squeeze(pred)
    
    num_labels = int(max(label.max(), pred.max()))
    
    plt.figure(figsize=(12, 12))
    plt.imshow(image, cmap='gray')
    
    if num_labels <= 1:
        # Binary case - use standard Green/Red
        gt_mask = np.zeros((*label.shape, 4))
        gt_mask[label > 0] = [0, 1, 0, alpha]
        pred_mask = np.zeros((*pred.shape, 4))
        pred_mask[pred > 0] = [1, 0, 0, alpha]
        plt.imshow(gt_mask)
        plt.imshow(pred_mask)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=alpha, label='GT'),
                           Patch(facecolor='red', alpha=alpha, label='Pred')]
    else:
        # Multi-class case - use distinct colors
        cmap = plt.get_cmap('tab10')
        # Overlay GT with contours per class
        for i in range(1, num_labels + 1):
            color = cmap(i % 10)
            if np.any(label == i):
                plt.contour(label == i, colors=[color], levels=[0.5], linewidths=2, linestyles='solid')
            if np.any(pred == i):
                plt.contour(pred == i, colors=[color], levels=[0.5], linewidths=2, linestyles='dashed')
        
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle='solid', label='GT Boundary'),
                           Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Pred Boundary')]
        for i in range(1, num_labels + 1):
            legend_elements.append(Patch(facecolor=cmap(i % 10), label=f'Class {i}'))

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_distribution(metrics_df, metric_names=None, save_path=None):
    """
    Plots the distribution of metrics using Violin plots.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metric results for multiple samples.
            Expects columns matching names in `metric_names`.
        metric_names (list[str], optional): List of column names to plot. Defaults to ['Dice', 'IoU'].
        save_path (str, optional): Path to save the figure if provided.
    """
    if metric_names is None:
        metric_names = ['Dice', 'IoU']
    plt.figure(figsize=(12, 6))
    melted_df = metrics_df.melt(value_vars=metric_names, var_name='Metric', value_name='Score')
    
    sns.violinplot(data=melted_df, x='Metric', y='Score', inner='box', palette="muted")
    plt.title("Metric Distribution")
    plt.ylim(0, 1.1) # Assuming normalized metrics
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_metric_correlation(metrics_df, x_metric='Dice', y_metric='HD95', save_path=None):
    """
    Plots the relationship between two metrics using a Scatter plot with a regression line.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing metric results.
        x_metric (str): Metric for the x-axis.
        y_metric (str): Metric for the y-axis.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(data=metrics_df, x=x_metric, y=y_metric, scatter_kws={'alpha':0.5})
    plt.title(f"Correlation: {x_metric} vs {y_metric}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_model_comparison(comparison_df, metrics=None, save_path=None):
    """
    Compares multiple models across several metrics using a Grouped Bar chart.

    Args:
        comparison_df (pd.DataFrame): DataFrame with columns ['Model', 'Metric', 'Score'].
            Or a wide format where 'Model' is a column or index.
        metrics (list[str], optional): Metrics to include in the comparison. Defaults to ['Dice', 'IoU'].
        save_path (str, optional): Path to save the figure if provided.
    """
    if metrics is None:
        metrics = ['Dice', 'IoU']
    # If comparison_df is not in long format, melt it
    if 'Metric' not in comparison_df.columns:
        # Assuming 'Model' is a column and the metrics are other columns
        melt_vars = [m for m in metrics if m in comparison_df.columns]
        id_vars = ['Model'] if 'Model' in comparison_df.columns else []
        comparison_df = comparison_df.melt(id_vars=id_vars, value_vars=melt_vars, var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_summary_report(metrics_df, overlay_info=None, save_path=None):
    """
    Creates a summary grid visualization.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics (Dice, IoU, HD95, etc.).
        overlay_info (dict, optional): Keys ['image', 'label', 'pred'] for an example overlay.
            - image: (H, W), (1, H, W), (3, H, W), or (H, W, 3)
            - label/pred: (H, W) or (1, H, W) binary masks.
        save_path (str, optional): Path to save the report if provided.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Metric Distribution (Box Plots)
    ax1 = fig.add_subplot(gs[0, 0])
    metric_cols = [c for c in ['Dice', 'IoU', 'Precision', 'Recall'] if c in metrics_df.columns]
    melted = metrics_df.melt(value_vars=metric_cols, var_name='Metric', value_name='Score')
    sns.boxplot(data=melted, x='Metric', y='Score', ax=ax1, palette='Set2')
    ax1.set_title("Metric Distribution Overview")
    ax1.set_ylim(0, 1.1)

    # 2. HD95 / ASD (Different scale)
    ax2 = fig.add_subplot(gs[0, 1])
    dist_cols = [c for c in ['HD95', 'ASD'] if c in metrics_df.columns]
    if dist_cols:
        melted_dist = metrics_df.melt(value_vars=dist_cols, var_name='Metric', value_name='Distance (mm)')
        sns.violinplot(data=melted_dist, x='Metric', y='Distance (mm)', ax=ax2, inner='point')
        ax2.set_title("Boundary Error Metrics")

    # 3. Correlation (Dice vs HD95)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'Dice' in metrics_df.columns and 'HD95' in metrics_df.columns:
        sns.scatterplot(data=metrics_df, x='Dice', y='HD95', ax=ax3, alpha=0.6)
        sns.regplot(data=metrics_df, x='Dice', y='HD95', ax=ax3, scatter=False, color='red')
        ax3.set_title("Dice vs HD95 Correlation")

    # 4. Example Overlay (if provided)
    ax4 = fig.add_subplot(gs[1, 1])
    if overlay_info:
        img = np.squeeze(overlay_info['image'])
        lbl = np.squeeze(overlay_info['label'])
        prd = np.squeeze(overlay_info['pred'])

        # Handle image shapes
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:  # (1, H, W) or (3, H, W)
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:  # (H, W, 1)
                img = np.squeeze(img, axis=-1)

        ax4.imshow(img, cmap='gray' if img.ndim == 2 else None)

        gt_mask = np.zeros((*lbl.shape, 4))
        gt_mask[lbl > 0.5] = [0, 1, 0, 0.4]
        ax4.imshow(gt_mask)

        pred_mask = np.zeros((*prd.shape, 4))
        pred_mask[prd > 0.5] = [1, 0, 0, 0.4]
        ax4.imshow(pred_mask)
        ax4.set_title("Example Segmentation Overlay")
        ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, "No Overlay Data Provided", ha='center', va='center')
        ax4.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_radar_chart(comparison_df, metrics=None, save_path=None):
    """
    Plots a Radar (Spider) chart comparing models across multiple metrics.
    Very common in SOTA papers to show balanced performance.

    Args:
        comparison_df (pd.DataFrame): DataFrame where 'Model' is a column or index.
        metrics (list[str], optional): Metrics to plot. Defaults to ['Dice', 'IoU', 'Precision', 'Recall'].
        save_path (str, optional): Path to save the figure if provided.
    """
    if metrics is None:
        metrics = ['Dice', 'IoU', 'Precision', 'Recall']
    from math import pi
    
    # Preprocess data: assume Model is index/column, metrics are columns
    if 'Model' in comparison_df.columns:
        comparison_df = comparison_df.set_index('Model')
    
    categories = metrics
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for model_name, row in comparison_df.iterrows():
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title("Model Performance Profile (Radar Chart)")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_dice_cdf(metrics_df, metric_name='Dice', save_path=None):
    """
    Plots the Cumulative Distribution Function (CDF) of a metric (e.g. Dice).
    Commonly used to show the percentage of cases above a certain threshold.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metric results.
        metric_name (str, optional): Metric name to plot. Defaults to 'Dice'.
        save_path (str, optional): Path to save the figure if provided.
    """
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(data=metrics_df, x=metric_name)
    plt.axvline(x=0.8, color='red', linestyle='--', label='80% Threshold')
    plt.title(f"Cumulative Distribution of {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.xlabel(f"{metric_name} Score")
    plt.ylabel("Proportion of Cases")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pixel_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plots a pixel-wise confusion matrix for a single sample.

    Args:
        y_true (np.ndarray | torch.Tensor): Ground truth labels. Expected shape (H, W).
        y_pred (np.ndarray | torch.Tensor): Predicted labels. Expected shape (H, W).
        labels (list[int/str], optional): Class labels or names. Defaults to [0, 1].
        save_path (str, optional): Path to save the figure if provided.
    """
    if labels is None:
        labels = [0, 1]
    from sklearn.metrics import confusion_matrix
    
    y_true_flat = np.squeeze(y_true).flatten()
    y_pred_flat = np.squeeze(y_pred).flatten()
    
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    # Normalize
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm.astype('float')), where=cm_sum!=0)
    
    # Define display labels
    display_labels = [str(lab) for lab in labels]
    if len(display_labels) == 2 and labels == [0, 1]:
        display_labels = ['BG', 'Target']
        
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=display_labels, yticklabels=display_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Pixel-wise Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def _get_binary_mask(m, idx):
    """Internal helper to extract a binary mask for a specific class."""
    m = np.squeeze(m)
    if m.ndim == 3:  # One-hot (C, H, W)
        if idx < 0 or idx >= m.shape[0]:
            raise ValueError(f"_get_binary_mask: class_index {idx} is out of bounds for tensor with {m.shape[0]} channels.")
        return m[idx] > 0.5
    # Single channel or already squeezed
    return m == idx if m.max() > 1 else m > 0.5

def plot_segmentation_error_heatmap(image, label, pred, class_index=1, save_path=None):
    """
    Visualizes True Positives (TP), False Positives (FP), and False Negatives (FN) for a specific class.

    Args:
        image (np.ndarray | torch.Tensor): Background image. 
            Expected shapes: (H, W), (1, H, W), (3, H, W), or (H, W, 3).
        label (np.ndarray | torch.Tensor): Ground truth mask. 
            Expected shapes: (H, W), (1, H, W), or (C, H, W).
        pred (np.ndarray | torch.Tensor): Predicted mask. 
            Expected shapes: (H, W), (1, H, W), or (C, H, W).
        class_index (int, optional): The class to analyze for errors. Defaults to 1.
            If masks are single-channel, this is ignored and binary classification is assumed.
        save_path (str, optional): Path to save the figure if provided.
    """
    if isinstance(image, torch.Tensor): image = image.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor): pred = pred.detach().cpu().numpy()

    # Handle image shapes
    if image.ndim == 3:
        if image.shape[0] in [1, 3]:  # (1, H, W) or (3, H, W)
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:  # (H, W, 1)
            image = np.squeeze(image, axis=-1)

    label_bin = _get_binary_mask(label, class_index)
    pred_bin = _get_binary_mask(pred, class_index)
    
    tp = np.logical_and(label_bin, pred_bin)
    fp = np.logical_and(np.logical_not(label_bin), pred_bin)
    fn = np.logical_and(label_bin, np.logical_not(pred_bin))

    # Normalize image
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    if img_norm.ndim == 2:
        overlay = np.stack([img_norm]*3, axis=-1)
    else:
        overlay = img_norm.copy()

    # TP: Green, FP: Red, FN: Blue
    overlay[tp] = [0, 0.8, 0]
    overlay[fp] = [0.8, 0, 0]
    overlay[fn] = [0, 0, 0.8]

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Error Analysis (Class {class_index}): TP (Green), FP (Red), FN (Blue)")
    plt.axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='True Positive'),
                       Patch(facecolor='red', label='False Positive'),
                       Patch(facecolor='blue', label='False Negative')]
    plt.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_boundary_comparison(label, pred, save_path=None):
    """
    Visualizes the boundaries of Ground Truth vs Prediction.

    Args:
        label (np.ndarray | torch.Tensor): Ground truth mask. Expected shape (H, W) or (1, H, W).
        pred (np.ndarray | torch.Tensor): Predicted mask. Expected shape (H, W) or (1, H, W).
        save_path (str, optional): Path to save the figure if provided.
    """
    from scipy.ndimage import binary_dilation
    
    def get_boundary(mask):
        dilated = binary_dilation(mask)
        return np.logical_and(dilated, np.logical_not(mask))

    label = np.squeeze(label) > 0.5
    pred = np.squeeze(pred) > 0.5
    
    gt_boundary = get_boundary(label)
    pred_boundary = get_boundary(pred)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(label, cmap='gray', alpha=0.1)
    plt.contour(gt_boundary, colors='green', levels=[0.5], linewidths=2)
    plt.contour(pred_boundary, colors='red', levels=[0.5], linewidths=2)
    
    plt.title("Boundary Comparison: GT (Green) vs Pred (Red)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_performance_vs_size(metrics_df, save_path=None):
    """
    Plots IoU as a function of object size (pixel count).

    Args:
        metrics_df (pd.DataFrame): DataFrame containing 'Size' and 'IoU' columns.
        save_path (str, optional): Path to save the figure if provided.
    """
    if 'Size' not in metrics_df.columns:
        # If size is not present, skip
        return
        
    plt.figure(figsize=(8, 6))
        
    sns.scatterplot(data=metrics_df, x='Size', y='IoU', alpha=0.6)
    plt.title("Segmentation Performance vs Object Size")
    plt.xlabel("Object Size (pixels)")
    plt.ylabel("IoU Score")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_history(history_df, metrics=None, save_path=None):
    """
    Plots Training vs Validation curves (Loss, Dice, etc.).

    Args:
        history_df (pd.DataFrame): DataFrame with columns like 'train_Loss', 'val_Loss', etc.
        metrics (list[str], optional): Metrics to include. Defaults to ['Loss', 'Dice'].
        save_path (str, optional): Path to save the figure if provided.
    """
    if metrics is None:
        metrics = ['Loss', 'Dice']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1: axes = [axes]
    
    for i, metric in enumerate(metrics):
        if f'train_{metric}' in history_df.columns:
            sns.lineplot(data=history_df, x=history_df.index, y=f'train_{metric}', ax=axes[i], label='Train')
        if f'val_{metric}' in history_df.columns:
            sns.lineplot(data=history_df, x=history_df.index, y=f'val_{metric}', ax=axes[i], label='Val')
        axes[i].set_title(f"{metric} over Epochs")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].legend()
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
