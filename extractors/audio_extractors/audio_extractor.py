# from settings import DEVICE
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
import numpy as np


class AudioEmbeddingExtractor:
    """
    Extracts embedding based on scene recognition task
    """

    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.sample_rate = 16000
        self.model  = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", output_hidden_states=True)

    def extract_base_feature(self, audio_path):
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(self.sample_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return self.feature_extractor(sound_array, sampling_rate=self.sample_rate,
                                      padding=True, return_tensors="pt")

    def extract_embedding(self, audio_path):
        input_audio = self.extract_base_feature(audio_path)
        model_output = self.model.forward(input_audio.input_values.float())
        hidden_states = model_output["hidden_states"]
        last_layer_hidden_states = hidden_states[
                24
            ]  # 25 = len(hidden_states) , dim = (batch_size, seq_len, 1024)
        cls_hidden_state = last_layer_hidden_states[:, 0, :]
        return cls_hidden_state
