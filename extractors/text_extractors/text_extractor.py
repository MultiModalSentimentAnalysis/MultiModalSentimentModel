from transformers import AutoTokenizer, pipeline
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

        self.device = device

        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.to_tensor = to_tensor

        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3, output_hidden_states=True
        ).to(self.device)

        # C1
        self.generator = pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def extract_embedding(
        self,
        input_batch_sentences,
    ):
        encoded_input = self.tokenizer(
            input_batch_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            hidden_states = model_output["hidden_states"]
            last_layer_hidden_states = hidden_states[
                12
            ]  # 12 = len(hidden_states) , dim = (batch_size, seq_len, 768)
            cls_hidden_state = last_layer_hidden_states[:, 0, :]

        return cls_hidden_state

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
