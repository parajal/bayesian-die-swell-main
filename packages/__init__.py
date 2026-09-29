import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from .main import ROMCurve4BayesianInference
__all__ = ["ROMCurve4BayesianInference"]
