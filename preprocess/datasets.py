import os, cv2, torch, ast
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import trange


class MSCTDDataSet(Dataset):
    """MSCTD dataset."""

    def __init__(
        self,
        base_path="data/",
        dataset_type="train",
        data_size=None,
        load=False,
        raw=False,
    ):
        """
        Args:
            base_path (str): path to data folder
            dataset_type (str): dev, train, test
        """
        base_path = Path(base_path)
        self.base_path = base_path
        self.save_path = base_path / "saved_features"
        self.dataset_type = dataset_type
        self.text_file_path = base_path / f"english_{dataset_type}.txt"
        self.seq_file_path = base_path / f"image_index_{dataset_type}.txt"
        self.sentiment_file_path = base_path / f"sentiment_{dataset_type}.txt"
        self.image_dir = base_path / "images" / dataset_type
        self.correct_indexes_file_path = (
            base_path / "correct_indexes" / f"correct_indexes_{dataset_type}.txt"
        )

        self.data_size = data_size
        self.load = load
        self.raw = raw

        self.texts = None
        self.sentiments = None
        self.indexes = None
        self.face_embeddings = None
        self.text_embeddings = None
        self.load_data()
        self.face_embedding_extractor = self.get_face_embedding_extractor()
        self.text_embedding_extractor = TextEmbeddingExtractor()

    def get_face_embedding_extractor(self):
        fd = FaceDetection("MTCNN", minimum_confidence=0.95)
        fa = FaceAlignment()
        fn = FaceNormalizer()
        model_name = "enet_b0_8_best_afew"
        fer = FaceEmotionRecognizer(device, model_name)
        fre = (
            FaceEmbeddingExtractor()
            .set_face_detection_model(fd)
            .set_face_alignment_model(fa)
            .set_face_normalizer_model(fn)
            .set_face_emotion_recognition_model(fer)
        )
        return fre

    def load_data(self):
        with open(self.text_file_path) as text_file, open(
            self.sentiment_file_path
        ) as sentiment_file, open(self.correct_indexes_file_path) as correct_file:
            texts = [t.strip() for t in text_file.readlines()]
            sentiments = [int(t.strip()) for t in sentiment_file.readlines()]
            face_embeddings = None
            text_embeddings = None
            corrects = [int(c.strip()) for c in correct_file.readlines()]
            if self.load:
                try:
                    face_embeddings = torch.load(
                        self.save_path / f"face_embeddings_{self.dataset_type}.pt"
                    )
                    text_embeddings = torch.load(
                        self.save_path / f"text_embeddings_{self.dataset_type}.pt"
                    )
                    corrects = torch.load(
                        self.save_path / f"real_indexes_{self.dataset_type}.pt"
                    )
                except Exception as e:
                    print(e)
                    print(
                        "Warning: passed load=True but not embedding file was located. Not loading"
                    )

            correct_texts = [texts[i] for i in corrects]
            correct_sentiments = [sentiments[i] for i in corrects]
        # with open(self.image_index_path) as f:
        #     images = [ast.literal_eval(t.strip()) for t in f.readlines()]

        if self.data_size:
            correct_texts = correct_texts[: self.data_size]
            correct_sentiments = correct_sentiments[: self.data_size]
            if face_embeddings:
                face_embeddings = face_embeddings[: self.data_size, :]
            if text_embeddings:
                face_embeddings = text_embeddings[: self.data_size, :]
            # images = images[: self.data_size]

        self.texts = correct_texts
        self.text_embeddings = text_embeddings
        self.sentiments = correct_sentiments
        self.indexes = corrects
        self.face_embeddings = face_embeddings

    def __len__(self):
        return len(self.texts)

    def get_face_features(self, index):
        if self.load and not self.face_embeddings is None:
            return self.face_embeddings[index]
        real_index = self.indexes[index]
        img_name = self.image_dir / f"{real_index}.jpg"
        image = cv2.imread(str(img_name))[:, :, ::-1]
        (
            predictions,
            scores,
            representations,
        ) = self.face_embedding_extractor.extract_representation(image)
        return representations[0]

    def get_sentiment(self, index):
        return self.sentiments[index]

    def get_text(self, index):
        if self.load and not self.text_embeddings is None:
            return self.text_embeddings[index]
        text = self.texts[index]
        text = self.text_embedding_extractor.extract_embedding([text])[0]
        return text

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        face_embedding = self.get_face_features(index)
        sentiment = self.get_sentiment(index)
        text = self.get_text(index)
        sample = {
            "real_index": self.indexes[index],
            "face_embedding": face_embedding,
            "text_embedding": text,
            "sentiment": sentiment,
        }
        if self.raw:
            sample["text"] = self.texts[index]
            # sample["image"] = pass

        return sample

    def save_features(self):
        face_embeddings = []
        text_embeddings = []
        real_indexes = []
        for i in trange(len(self)):
            record = self[i]
            face_embeddings.append(record["face_embedding"])
            text_embeddings.append(record["text_embedding"])
            real_indexes.append(record["real_index"])
            if i == 50:
                break
        face_embeddings = torch.stack(face_embeddings, dim=0)
        print(face_embeddings.shape)
        torch.save(
            face_embeddings, self.save_path / f"face_embeddings_{self.dataset_type}.pt"
        )
        text_embeddings = torch.stack(text_embeddings, dim=0)
        print(text_embeddings.shape)
        torch.save(
            text_embeddings, self.save_path / f"text_embeddings_{self.dataset_type}.pt"
        )
        real_indexes = torch.tensor(real_indexes)
        print(real_indexes.shape)
        torch.save(
            real_indexes, self.save_path / f"real_indexes_{self.dataset_type}.pt"
        )


# COMPLETE
if __name__ == "__main__":
    dataset = MSCTDDataSet("data/", "val")
    print(dataset[0])
