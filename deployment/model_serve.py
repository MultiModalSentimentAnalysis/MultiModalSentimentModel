import torch
import cv2
import numpy as np

# deployment imports/
from fastapi import FastAPI, UploadFile
import ray
from ray import serve

# local imports
from extractors.face_extractors.face_extractor import FaceEmbeddingExtractor
from extractors.pose_extractors.pose_extractor import PoseEmbeddingExtractor
from extractors.text_extractors.english_text_extractor import TextEmbeddingExtractor


app = FastAPI()
FACE_EMBEDDING_SIZE = 1280
ENG_TEXT_EMBEDDING_SIZE = 768
GER_TEXT_EMBEDDING_SIZE = 768
POSE_EMBEDDING_SIZE = 34
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@serve.deployment(
    _autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 5,
    },
    version="v1",
)
@serve.ingress(app)
class SentimentAnalyser:
    def __init__(self, model_path="./models/eng_model.pt", language="English"):
        self.base_model = torch.load(model_path)
        self.language = language
        self.face_embbedder = self.load_face_embedder()
        self.text_embedder = self.load_text_embedder()
        self.pose_embedder = self.load_pose_embedder()

    def load_face_embedder(self):
        return FaceEmbeddingExtractor()

    def load_text_embedder(self):
        if self.language == "eng":
            return TextEmbeddingExtractor()

    def load_pose_embedder(self):
        return PoseEmbeddingExtractor()

    def get_face_embedding(self, image):
        (
            predictions,
            scores,
            representations,
        ) = self.face_embedding_extractor.extract_embedding(image)
        return representations

    def get_pose_embedding(self, image):
        return self.pose_embedding_extractor.extract_embedding(image)

    def get_image_embeddings(self, image):
        face_embedding = self.get_face_embedding(image)
        pose_embedding = self.get_pose_embedding(image)
        return face_embedding, pose_embedding

    def get_text_embedding(self, text):
        embedding = self.text_embedder.extract_embedding([text])[0]
        return embedding

    def preprocess_data(self, txt, img):
        try:
            face_embedding, pose_embedding = self.get_image_embeddings(img)
            # scene_embedding = self.get_scene_embedding(index) # comment
        except Exception as e:
            print(e)
            face_embedding = torch.ones(FACE_EMBEDDING_SIZE).to(DEVICE) * -123
            pose_embedding = torch.ones(POSE_EMBEDDING_SIZE).to(DEVICE) * -123
        text_embedding = self.get_text_embedding(txt)
        return {
            "text_embedding": [text_embedding],
            "pose_embedding": [pose_embedding],
            "face_embedding": [face_embedding],
        }

    @app.post("/single/")
    async def get_sentiment(self, text: str, img: UploadFile):
        contents = await img.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_data = self.preprocess_data(text, img)
        inputs = torch.cat(
            (
                processed_data["face_embedding"],
                processed_data["text_embedding"],
                processed_data["pose_embedding"],
            ),
            1,
        )
        logits = self.base_model(inputs)
        outputs = logits.argmax(dim=1)
        return {"output": outputs}


def register_task():
    ray.init(address="auto", namespace="serve")
    serve.start(detached=True, http_options={"host": "0.0.0.0"})
    SentimentAnalyser.deploy()
