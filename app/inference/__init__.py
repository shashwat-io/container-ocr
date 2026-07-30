from flask import Flask
from app.config import get_config


class InferenceService:
    def __init__(self, app):
        self.app = app
        self._inferer = None
        self.config = get_config()

    @classmethod
    def init_app(cls, app):
        return cls(app)

    def get_inferer(self):
        if self._inferer is None:
            from app.inference.predictor import ContainerPredictor
            self._inferer = ContainerPredictor()
        return self._inferer

    def predict(self, image_bytes, filename=None):
        inferer = self.get_inferer()
        return inferer.predict(image_bytes, filename)