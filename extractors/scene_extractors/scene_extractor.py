from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from settings import DEVICE


class SceneEmbeddingExtractor:
    """
    Extracts embedding based on scene recognition task
    """

    def __init__(self):
        self.weights = ResNet50_Weights.DEFAULT
        self.pretrained_model = resnet50(weights=self.weights)
        self.pretrained_model.eval().to(DEVICE)
        self.preprocess = self.weights.transforms()
        self.feature_extractor = self.remove_last_layer()

    def remove_last_layer(self):
        modules = list(self.pretrained_model.children())[:-1]
        model = nn.Sequential(*modules)
        model.eval()
        return model

    def extract_embedding(self, image):
        transformed_image = self.preprocess(image).unsqueeze(0)
        return self.feature_extractor(transformed_image).ravel()
