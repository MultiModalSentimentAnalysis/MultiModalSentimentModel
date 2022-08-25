import torch, torchvision, cv2
from torchvision.transforms import transforms as transforms
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights


class PoseEmbeddingExtractor:
    def __init__(self, device="cpu"):
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, num_keypoints=17
        ).to(device)
        self.model.eval()
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor()])

    def extract_embedding(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)
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
