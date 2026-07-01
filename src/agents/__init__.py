# AI agents for prediction, optimization, and anomaly detection
# Lazy imports so missing optional deps (tensorflow, prophet) don't crash the package

try:
    from .predictive_agent import PredictiveAgent
    __all__ = ['PredictiveAgent']
except ImportError:
    pass  # tensorflow/prophet not installed; PredictiveAgent unavailable

try:
    from .anomaly_agent import AnomalyAgent
    __all__ = [*globals().get('__all__', []), 'AnomalyAgent']
except ImportError:
    pass