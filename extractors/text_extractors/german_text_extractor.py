import torch
import re
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from typing import List
from settings import DEVICE


class GermanTextEmbeddingExtractor:
    """
    Extracts embedding of the text using [CLS] token of a Bert based model.
    """

    def __init__(
        self,
        model_name="oliverguhr/german-sentiment-bert",
        max_length=128,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, output_hidden_states=True
        ).to(DEVICE)

        self.generator = pipeline(
            task="sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
        )
        self.clean_chars = re.compile(r"[^A-Za-züöäÖÜÄß ]", re.MULTILINE)
        self.clean_http_urls = re.compile(r"https*\S+", re.MULTILINE)
        self.clean_at_mentions = re.compile(r"@\S+", re.MULTILINE)

    def predict_sentiment(self, texts: List[str]) -> List[str]:
        texts = [self.clean_text(text) for text in texts]
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        # truncation=True limits number of tokens to model's limitations (512)
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = encoded.to(DEVICE)
        with torch.no_grad():
            logits = self.model(**encoded)

        label_ids = torch.argmax(logits[0], axis=1)
        return [self.model.config.id2label[label_id.item()] for label_id in label_ids]

    def replace_numbers(self, text: str) -> str:
        return (
            text.replace("0", " null")
            .replace("1", " eins")
            .replace("2", " zwei")
            .replace("3", " drei")
            .replace("4", " vier")
            .replace("5", " fünf")
            .replace("6", " sechs")
            .replace("7", " sieben")
            .replace("8", " acht")
            .replace("9", " neun")
        )

    def clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub("", text)
        text = self.clean_at_mentions.sub("", text)
        text = self.replace_numbers(text)
        text = self.clean_chars.sub("", text)  # use only text chars
        text = " ".join(
            text.split()
        )  # substitute multiple whitespace with single whitespace
        text = text.strip().lower()
        return text

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
        return self.predict_sentiment(input_batch_sentences)
