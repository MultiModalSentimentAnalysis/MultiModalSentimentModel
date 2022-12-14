import torch
from settings import DEVICE
from transformers import pipeline, RobertaForSequenceClassification, AutoTokenizer


class EnglishTextEmbeddingExtractor:
    """
    Extracts embedding of the text using [CLS] token of a Roberta based model.
    """

    def __init__(
        self,
        model_name="pysentimiento/robertuito-sentiment-analysis",
        max_length=128,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3, output_hidden_states=True
        ).to(DEVICE)

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
        ).to(DEVICE)

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
