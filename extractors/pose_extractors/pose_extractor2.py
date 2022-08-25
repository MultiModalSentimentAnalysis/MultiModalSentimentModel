import torch
import numpy as np
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord


class PoseEmbeddingExtractor:
    def __init__(
        self,
    ):
        self.detector = model_zoo.get_model("yolo3_mobilenet1.0_coco", pretrained=True)
        self.pose_net = model_zoo.get_model("simple_pose_resnet18_v1b", pretrained=True)
        self.detector.reset_class(["person"], reuse_weights=["person"])

    def detect_person(self, x, image):
        class_IDs, scores, bounding_boxs = self.detector(x)
        pose_input, upscale_bbox = detector_to_simple_pose(
            image, class_IDs, scores, bounding_boxs
        )
        return pose_input, upscale_bbox

    def get_most_confident_coords(self, predicted_heatmap, upscale_bbox):
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        mean_confidence = np.mean(confidence[:, :, 0].asnumpy(), axis=1)
        best_confidence_arg = mean_confidence.argmax()
        best_coords = pred_coords[best_confidence_arg].asnumpy().ravel()
        return best_coords

    def extract_embedding(self, image_path):
        x, image = data.transforms.presets.ssd.load_test(image_path, short=512)
        pose_input, upscale_bbox = self.detect_person(x, image)
        predicted_heatmap = self.pose_net(pose_input)
        best_coords = self.get_most_confident_coords(predicted_heatmap, upscale_bbox)
        return torch.tensor(best_coords)
