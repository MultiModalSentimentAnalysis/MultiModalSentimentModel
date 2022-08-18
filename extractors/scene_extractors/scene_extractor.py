from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from torch import nn


class SceneRepresentationExtractor:

    def __init__(self, pretrained_model, weights):
        self.weights = weights
        self.preprocess = weights.transforms()
        self.feature_extractor = self.remove_last_layer(pretrained_model)

    @staticmethod
    def remove_last_layer(pretrained_model):
        modules = list(pretrained_model.children())[:-1]
        model = nn.Sequential(*modules)
        model.eval()
        return model

    def extract_representation(self, image):
        transformed_image = self.preprocess(image).unsqueeze(0)
        return self.feature_extractor(transformed_image).ravel()


