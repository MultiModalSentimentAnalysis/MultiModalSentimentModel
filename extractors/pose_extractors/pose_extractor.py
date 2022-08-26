import torch, torchvision
from torchvision.transforms import transforms as transforms
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from settings import DEVICE


class PoseEmbeddingExtractor:
    """
    Extracts embedding based on pose of the persons in the image. Each person is consisted of 17 keypoints
    and they are used as a feature.
    """

    def __init__(self):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, num_keypoints=17
        ).to(DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def extract_embedding(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(image)

        keypoints_scores = outputs[0]["keypoints_scores"]
        best_score = torch.mean(keypoints_scores, axis=1).argmax().item()
        keypoints = outputs[0]["keypoints"][best_score, :, :2]
        return keypoints.ravel()


# p = PoseEmbeddingExtractor(device=device)
# path = 'data/images/val/4965.jpg'
# img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
# p.extract_embedding(img).shape
