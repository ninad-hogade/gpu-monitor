"""
gpu_monitor - A Python package for real-time GPU and system resource monitoring.

Modules:
- gpu_metrics: Functions for collecting GPU-specific metrics.
- measure: High-level monitoring and execution utilities.
- utils: Helper functions for file handling and logging.
"""

from .gpu_metrics import initialize_nvml, shutdown_nvml, collect_gpu_metrics_nvml
from .measure import measure_metrics
from .utils import (
    log_gpu_metrics,
    report_summary_metrics,
    monitor_resource_usage,
    generate_log_filename,
)

__version__ = "1.0.0"
__author__ = "Your Name"
