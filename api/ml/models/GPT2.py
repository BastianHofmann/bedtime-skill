import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

from .model import Model

class GPT2(Model):
    def __init__(self, model_id: str = None, server: str = None, parameters: dict = None):
        # self._model_path = Path(__file__).parent / "weigths" / f"{model_id}.h5"
        super().__init__(model_id, server, parameters)

    def train(self):
        self._model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self._tokenizer.eos_token_id)
        return self

    def predict(self, X: str) -> np.ndarray:
        print('predicting')
        input_ids = self._tokenizer.encode(X, return_tensors='tf')

        greedy_output = self._model.generate(
            input_ids,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True)

        return self._tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    def save(self):
        if self._model is not None:
            print('saving')
            # would be the case in tf
            # self._model.save_weights(self._model_path)
            self._model.save_pretrained(self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # check if model exists
        if os.path.exists(self._model_path):
            print('loading')
            # https://huggingface.co/transformers/training.html#fine-tuning-in-native-tensorflow-2
            self._model = TFGPT2LMHeadModel.from_pretrained(self._model_path)
            # self._model.load_weights(self._model_path)
            
        return self

if __name__ == "__main__":
    # execute file with python -m models.GPT2
    model = GPT2(model_id="TX4", server="http://ec2-18-193-73-211.eu-central-1.compute.amazonaws.com:8000")
    # model.train()
    # model.save()
    model.save_to_server(save_files=True)
    # add prediction
    pre = model.predict("The man went skiing")
    print(pre)
    model.upload_prediction("skiing", pre)
