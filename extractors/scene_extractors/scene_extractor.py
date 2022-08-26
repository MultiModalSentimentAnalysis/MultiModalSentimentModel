from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
class SceneEmbeddingExtractor:

    def __init__(self, device = 'cpu'):
        self.weights = ResNet50_Weights.DEFAULT
        self.pretrained_model = resnet50(weights=self.weights)
        self.pretrained_model.eval().to(device)
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
