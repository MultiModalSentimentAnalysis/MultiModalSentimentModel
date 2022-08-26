import cv2, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from settings import (
    DEVICE,
    FACE_EMBEDDING_SIZE,
    POSE_EMBEDDING_SIZE,
    TEXTS_DIR,
    LABELS_DIR,
    IMAGES_DIR,
    SAVE_DIR,
)
from extractors.face_extractors.face_extractor import FaceEmbeddingExtractor
from extractors.pose_extractors.pose_extractor import PoseEmbeddingExtractor
from extractors.scene_extractors.scene_extractor import SceneEmbeddingExtractor
from extractors.text_extractors.english_text_extractor import (
    EnglishTextEmbeddingExtractor,
)
from extractors.text_extractors.german_text_extractor import (
    GermanTextEmbeddingExtractor,
)


class MSCTDDataSet(Dataset):
    """
    MSCTD dataset.

    It can be used with raw data, to extract embeddings or with saved features.
    """

    def __init__(
        self,
        split="train",
        data_size=None,
        load=False,
    ):
        """
        Args:
            split (str): val, train, test.
            data_size (int): None for full dataset. If provided dataset size will be reduced to data_size.
            load (bool): If false, all embeddings will be extracted and dataset works with bare text and image. If true, it loads all pre extracted embeddings.
                         Warning: don't use load=False for training. Always try using with load=True for training, to speedup process.
        """

        self.split = split
        self.eng_text_file_path = TEXTS_DIR / "english" / f"{split}.txt"
        self.ger_text_file_path = TEXTS_DIR / "german" / f"{split}.txt"
        self.sentiment_file_path = LABELS_DIR / f"sentiment_{split}.txt"
        self.image_dir = IMAGES_DIR / split

        self.data_size = data_size
        self.load = load

        self.eng_texts = None
        self.sentiments = None
        self.indexes = None
        self.face_embeddings = None
        self.pose_embeddings = None
        self.eng_text_embeddings = None
        self.load_data()
        self.face_embedding_extractor = FaceEmbeddingExtractor()
        self.eng_text_embedding_extractor = EnglishTextEmbeddingExtractor()
        self.ger_text_embedding_extractor = GermanTextEmbeddingExtractor()
        self.pose_embedding_extractor = PoseEmbeddingExtractor()
        self.scene_embedding_extractor = SceneEmbeddingExtractor()

    # text is not valid
    def load_data(self):
        if self.load:
            eng_texts = None
            ger_texts = None
            indexes = torch.load(SAVE_DIR / f"indexes_{self.split}.pt").to(DEVICE)
            sentiments = torch.load(SAVE_DIR / f"sentiments_{self.split}.pt").to(DEVICE)
            face_embeddings = torch.load(
                SAVE_DIR / f"face_embeddings_{self.split}.pt"
            ).to(DEVICE)
            pose_embeddings = torch.load(
                SAVE_DIR / f"pose_embeddings_{self.split}.pt"
            ).to(DEVICE)
            eng_text_embeddings = torch.load(
                SAVE_DIR / f"eng_text_embeddings_{self.split}.pt"
            ).to(DEVICE)
            ger_text_embeddings = torch.load(
                SAVE_DIR / f"ger_text_embeddings_{self.split}.pt"
            ).to(DEVICE)
            scene_embeddings = torch.load(
                SAVE_DIR / f"scene_embeddings_{self.split}.pt"
            ).to(DEVICE)

            assert (
                face_embeddings.shape[0] == pose_embeddings.shape[0]
            ), "ERROR:  face and pose list are not the same size in loading"
            assert (
                pose_embeddings.shape[0] == eng_text_embeddings.shape[0]
            ), "ERROR: text and pose list are not the same size in loading"
            assert (
                eng_text_embeddings.shape[0] == indexes.shape[0]
            ), "ERROR: text and index list are not the same size in loading"
            assert (
                indexes.shape[0] == sentiments.shape[0]
            ), "ERROR: index and sentiment list are not the same size in loading"

            print(face_embeddings.shape)
            print(pose_embeddings.shape)
            print(eng_text_embeddings.shape)
            print(ger_text_embeddings.shape)
            print(indexes.shape)
            print(sentiments.shape)

        else:
            with open(self.eng_text_file_path) as eng_text_file, open(
                self.ger_text_file_path
            ) as ger_text_file, open(self.sentiment_file_path) as sentiment_file:
                sentiments = [int(t.strip()) for t in sentiment_file.readlines()]
                eng_texts = [t.strip() for t in eng_text_file.readlines()]
                ger_texts = [t.strip() for t in ger_text_file.readlines()]
                indexes = range(len(sentiments))
                # indexes = torch.load(SAVE_DIR / f"indexes_{self.split}.pt").to(DEVICE) #WARNING: REMOVE

                face_embeddings = None
                pose_embeddings = None
                eng_text_embeddings = None
                ger_text_embeddings = None
                scene_embeddings = None

        if self.data_size:
            indexes = indexes[: self.data_size]
            sentiments = sentiments[: self.data_size]
            if not eng_texts is None:
                eng_texts = eng_texts[: self.data_size]
            if not ger_texts is None:
                ger_texts = ger_texts[: self.data_size]
            if not face_embeddings is None:
                face_embeddings = face_embeddings[: self.data_size, :]
            if not pose_embeddings is None:
                pose_embeddings = pose_embeddings[: self.data_size, :]
            if not scene_embeddings is None:
                scene_embeddings = scene_embeddings[: self.data_size, :]
            if not eng_text_embeddings is None:
                eng_text_embeddings = eng_text_embeddings[: self.data_size, :]

        self.eng_texts = eng_texts
        self.ger_texts = ger_texts
        self.sentiments = sentiments
        self.indexes = indexes
        self.face_embeddings = face_embeddings
        self.pose_embeddings = pose_embeddings
        self.eng_text_embeddings = eng_text_embeddings
        self.ger_text_embeddings = ger_text_embeddings
        self.scene_embeddings = scene_embeddings

    def __len__(self):
        return len(self.indexes)

    def get_face_embedding(self, image):
        (
            predictions,
            scores,
            representations,
        ) = self.face_embedding_extractor.extract_embedding(image)
        return representations

    def get_pose_embedding(self, image):
        return self.pose_embedding_extractor.extract_embedding(image)

    def get_image_embeddings(self, index):
        if self.load:
            return self.face_embeddings[index], self.pose_embeddings[index]

        image_name = self.image_dir / f"{index}.jpg"
        image = cv2.cvtColor(cv2.imread(str(image_name)), cv2.COLOR_BGR2RGB)
        face_embedding = self.get_face_embedding(image)
        pose_embedding = self.get_pose_embedding(image)
        return face_embedding, pose_embedding

    def get_scene_embedding(self, index):
        if self.load:
            return self.scene_embeddings[index]
        real_index = self.indexes[index]
        image_name = str(self.image_dir) + f"/{real_index}.jpg"
        image = read_image(image_name).to(DEVICE)
        return self.scene_embedding_extractor.extract_embedding(image)

    def get_sentiment(self, index):
        return self.sentiments[index]

    def get_eng_text(self, index):
        if self.load:
            return self.eng_text_embeddings[index]
        text = self.eng_texts[index]
        text = self.eng_text_embedding_extractor.extract_embedding([text])[0]
        return text

    def get_ger_text(self, index):
        if self.load:
            return self.ger_text_embeddings[index]
        text = self.ger_texts[index]
        text = self.ger_text_embedding_extractor.extract_embedding([text])[0]
        return text

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        try:
            face_embedding, pose_embedding = self.get_image_embeddings(index)
            scene_embedding = self.get_scene_embedding(index)
        except Exception as e:
            print(f"error for split:{self.split} index: {index}")
            print(e)
            face_embedding = torch.ones(FACE_EMBEDDING_SIZE).to(DEVICE) * -123
            pose_embedding = torch.ones(POSE_EMBEDDING_SIZE).to(DEVICE) * -123

        sentiment = self.get_sentiment(index)
        eng_text_embedding = self.get_eng_text(index)
        ger_text_embedding = self.get_ger_text(index)

        sample = {
            "index": self.indexes[index],
            "pose_embedding": pose_embedding,
            "face_embedding": face_embedding,
            "scene_embedding": scene_embedding,
            "eng_text_embedding": eng_text_embedding,
            "ger_text_embedding": ger_text_embedding,
            "sentiment": sentiment,
        }
        return sample


def get_dataset_and_dataloder(split, batch_size, data_size=None):
    dataset = MSCTDDataSet(split, data_size=data_size, load=True)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset, dataloader
