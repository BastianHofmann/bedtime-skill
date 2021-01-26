import math
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

from .model import Model
from .config import ML_DATASETS_PATH

class GPT2Tuned(Model):
    def __init__(self, model_id: str = None, server: str = None, parameters: dict = None):
        super().__init__(model_id, server, parameters)
        self._block_size = 128

    def tokenize(self, examples):
        return self._tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // self._block_size) * self._block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self._block_size] for i in range(0, total_length, self._block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self):
        self._model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=self._tokenizer.eos_token_id)
        datasets = load_dataset('text', data_files={'train':  ML_DATASETS_PATH / 'train_dataset.txt', 'validation': ML_DATASETS_PATH / 'val_dataset.txt'})

        print("Train dataset length: "+str(len(datasets["train"])))
        print("Validation dataset length: "+ str(len(datasets["validation"])))

        tokenized_datasets = datasets.map(self.tokenize, batched=True, num_proc=4, remove_columns=["text"])

        lm_datasets = tokenized_datasets.map(
            self.group_texts,
            batched=True,
            batch_size=1000,
            num_proc=4,
        )

        print(lm_datasets["train"][0])

        training_args = TrainingArguments(
            "test-clm",
            evaluation_strategy="epoch",
            learning_rate=self._parameters.get("learning_rate",  5e-5),
            num_train_epochs=self._parameters.get("num_train_epochs", 3),
            logging_steps=self._parameters.get("logging_steps", 500),
            save_steps=self._parameters.get("save_steps", 500),
            save_total_limit=self._parameters.get("save_total_limit", None),
        )

        print(str(self._model))

        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["validation"],
        )

        print("start training ------------------------")

        self._trainer.train()

        eval_results = self._trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        return self

    def predict(self, X: str, max_length: int = 50) -> np.ndarray:
        input_ids = self._tokenizer.encode(X, return_tensors="pt")

        greedy_output = self._model.generate(
            input_ids,
            max_length=max_length,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True)

        return self._tokenizer.decode(greedy_output[0], skip_special_tokens=True)

    def save(self):
        if self._model is not None and self._trainer is not None:
            print('saving')
            self._trainer.save_model(self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        if os.path.exists(self._model_path):
            print('loading')
            self._model = AutoModelForCausalLM.from_pretrained(self._model_path, pad_token_id=self._tokenizer.eos_token_id)

        return self

if __name__ == "__main__":
    # execute file with python -m models.GPT2Tuned
    model = GPT2Tuned(model_id="NY4", server="http://ec2-18-193-73-211.eu-central-1.compute.amazonaws.com:8000")
    model.train()
    model.predict("Once upon a time there was a horse in the mountains")
    # model.save()
    # model.save_to_server(save_files=True)
