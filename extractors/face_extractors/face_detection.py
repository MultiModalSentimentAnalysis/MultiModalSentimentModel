from facenet_pytorch import MTCNN
from settings import DEVICE


class FaceDetection:
    """
    Detects faces in the image using MTCNN netork.
    """

    def __init__(self, model_name, minimum_confidence):

        self.detected_faces_information = None
        self.model_name = model_name
        self.minimum_confidence = minimum_confidence
        if model_name == "MTCNN":
            detector_model = MTCNN(device=DEVICE)
            self.detect_faces_function = lambda input_image: detector_model.detect(
                input_image, landmarks=True
            )

    def extract_faces(self, input_image, return_detections_information=True):
        self.detect_faces__(input_image)
        faces = self.get_faces__(
            input_image,
        )
        if return_detections_information:
            return faces, self.detected_faces_information

        else:
            return faces

    def detect_faces__(self, input_image):
        detections = self.detect_faces_function(input_image)
        detections = [
            {
                "box": detections[0][i],
                "confidence": detections[1][i],
                "keypoints": {
                    "left_eye": detections[2][i][0],
                    "right_eye": detections[2][i][1],
                    "nose": detections[2][i][2],
                    "mouth_left": detections[2][i][3],
                    "mouth_right": detections[2][i][4],
                },
            }
            for i in range(detections[0].shape[0])
        ]
        self.detected_faces_information = list(
            filter(
                lambda element: element["confidence"] > self.minimum_confidence,
                detections,
            )
        )

    def get_detected_faces_information(self):
        return self.detected_faces_information

    def get_keypoints(
        self,
    ):
        return list(
            map(lambda element: element["keypoints"], self.detected_faces_information)
        )

    def get_faces__(
        self,
        input_image,
    ):
        boxes = [
            detection_information["box"]
            for detection_information in self.detected_faces_information
        ]
        y1y2x1x2 = [(int(y), int(y2), int(x), int(x2)) for x, y, x2, y2 in boxes]
        faces = [input_image[y1:y2, x1:x2] for y1, y2, x1, x2 in y1y2x1x2]
        return faces

    def get_eyes_coordinates(
        self,
    ):
        eyes_coordinates = [
            (info["keypoints"]["left_eye"], info["keypoints"]["right_eye"])
            for info in self.detected_faces_information
        ]
        return eyes_coordinates
