from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin


def build_model() -> ClassifierMixin:
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    return model
