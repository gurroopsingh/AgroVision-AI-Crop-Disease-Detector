"""
AgroVision inference: single-image prediction from trained checkpoints.
"""
from .predict import load_model, predict_image

__all__ = ["load_model", "predict_image"]
