"""
Utility modules for SAND Task 1 pipeline
"""

from .audio import (
    load_audio,
    normalize_audio,
    preemphasis,
    remove_silence,
    augment_audio,
    pad_or_truncate
)

from .metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance
)

from .visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_feature_importance,
    plot_training_curves,
    generate_results_summary,
    save_metrics_json
)

__all__ = [
    # Audio utilities
    "load_audio",
    "normalize_audio",
    "preemphasis",
    "remove_silence",
    "augment_audio",
    "pad_or_truncate",

    # Metrics
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_feature_importance",

    # Visualization
    "plot_per_class_metrics",
    "plot_training_curves",
    "generate_results_summary",
    "save_metrics_json",
]