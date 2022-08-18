from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import RobertaForSequenceClassification
import torch
import pickle


class TextEmbeddingExtractor:
    def __init__(
        self,
        model_name="pysentimiento/robertuito-sentiment-analysis",
        batch_size=250,
        show_progress_bar=True,
        to_tensor=True,
        max_length=128,
    ):
        self.model_name = model_name

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.to_tensor = to_tensor

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = RobertaForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        ).to(self.device)
        # self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.generator = pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def extract_embedding(
        self,
        encoded_input,
    ):
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        return sentence_embeddings

    def get_labels(self, input_batch_sentences):
        return self.generator(input_batch_sentences)

    @staticmethod
    def store_embeddings(file, embeddings):
        with open(file, "wb") as file_out:
            pickle.dump(
                {"embeddings": embeddings}, file_out, protocol=pickle.HIGHEST_PROTOCOL
            )

    @staticmethod
    def load_embeddings(file):
        with open(file, "rb") as file_in:
            stored_data = pickle.load(file_in)
            stored_embeddings = stored_data["embeddings"]

        return stored_embeddings
