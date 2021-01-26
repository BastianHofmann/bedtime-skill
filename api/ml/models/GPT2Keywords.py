import math
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (GPT2LMHeadModel, GPT2TokenizerFast, TrainingArguments) #GPT2TokenizerFast

from .config import ML_DATASETS_PATH
from .KeywordTrainer import KeywordTrainer
from .model import Model


class GPT2Keywords(Model):
    def __init__(self, model_id: str = None, server: str = None, parameters: dict = {}):
        super().__init__(model_id, server, parameters)
        # seq_len
        self._block_size = self._parameters.get("block_size", 200)
        self._labels_same_as_input = self._parameters.get("labels_same_as_input", False)
        # mask nouns in training in order to compute the loss with the keywords
        self._mask_nouns = self._parameters.get("mask_nouns", True)
        self._keyword_loss_weight = self._parameters.get("keyword_loss_weight", 1e-4) 

    def tokenize(self, examples):
        # tokenize words
        keys = ["keyword1", "keyword2", "keyword3"]
        if self._parameters.get("add_space", False):
            keywords = [ f" {examples[key]}" if examples[key] != self._special_word else examples[key] for key in keys]
            keywords = [self.get_token(keyword) for keyword in keywords]
        else:
            keywords = [self.get_token(examples[key]) for key in keys]
        
        result = self._tokenizer(examples["text"])
        result["keywords"] = keywords
        return result

    def assign_keywords(self, examples):
        input_ids = []
        labels = []
        attention_mask = []
        keywords = []
        for index, result in enumerate(examples['input_ids']):
            # result are the input_ids
            number_tokens = len(result)
            number_of_batches = number_tokens // self._block_size

            # decide to keep rest or truncate
            if number_of_batches - (number_of_batches * self._block_size) > 0.5 * self._block_size:
                number_of_batches += 1

           
            for i in range(number_of_batches):
                # the keywords plus the text 
                input_ids.append(examples["keywords"][index] + [self._special_token] + result[i*self._block_size:(i+1)*self._block_size])
            
                # same labels as input?
                if self._labels_same_as_input:
                    labels.append(input_ids[i].copy())
                else:
                    # for the labels we just add padding at te beginning
                    padding = [self._special_token] * 4
                    labels.append(padding + result[i*self._block_size:(i+1)*self._block_size])

                # attention mask just pad with one in the beginning
                attention_mask.append([1] * 4 + examples['attention_mask'][index][i*self._block_size:(i+1)*self._block_size])
            keywords += [examples["keywords"][index]] * number_of_batches  

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "keywords": keywords
        }

    def train(self):
        self._model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self._tokenizer.eos_token_id)
        self._model.resize_token_embeddings(len(self._tokenizer))

        output_name = "fairytales" 
        if self._parameters.get("only_grimm", False):
            output_name = "grimm_fairytales"

        train_path = ML_DATASETS_PATH / f'{output_name}_train.csv'
        val_path = ML_DATASETS_PATH / f'{output_name}_val.csv'
        datasets = load_dataset('csv', delimiter=";", data_files={'train': train_path, 'validation': val_path})

        print("Train dataset length: "+str(len(datasets["train"])))
        print("Validation dataset length: "+ str(len(datasets["validation"])))

        tokenized_datasets = datasets.map(self.tokenize, batched=False,
        remove_columns=["text", "keyword1", "keyword2", "keyword3", "title"]) # drop all unnecessary columns

        # have to use bached in order to change the output batch size
        final_datasets = tokenized_datasets.map(self.assign_keywords, batched=True, batch_size=5) # num_proc=4,

        # set the format to pytorch tensors
        final_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'keywords'])
        #print(final_datasets["train"][0])
        
        # print(self._tokenizer.convert_ids_to_tokens(final_datasets["train"][40]["keywords"]))
        # print(self._tokenizer.decode(final_datasets["train"][40]["input_ids"]))

        training_args = TrainingArguments(
            "test-clm",
            evaluation_strategy="epoch",
            learning_rate=self._parameters.get("learning_rate",  5e-5),
            num_train_epochs=self._parameters.get("num_train_epochs", 3),
            logging_steps=self._parameters.get("logging_steps", 500),
            save_steps=self._parameters.get("save_steps", 500),
            save_total_limit=self._parameters.get("save_total_limit", None),
            remove_unused_columns = False, # important for keywords to be passed to compute loss
            load_best_model_at_end = self._parameters.get("load_best_model",  False) 
        )

        self._trainer = KeywordTrainer(
            model=self._model,
            args=training_args,
            tokenizer=self._tokenizer,
            train_dataset=final_datasets["train"],
            eval_dataset=final_datasets["validation"],
            special_token= self._special_token,
            mask_nouns= self._mask_nouns,
            keyword_loss_weight= self._keyword_loss_weight,
            cos_similarity= self._parameters.get("cos_similarity",  False)
        )

        print("start training ------------------------")

        self._trainer.train()

        eval_results = self._trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        return self

    def predict(self, X: str, keywords: list = None,  max_length: int = 50) -> np.ndarray:
        if not keywords:
            raise Exception("Keywords have to be provided as list")
        
        tokens = []
        # check on which device the model is at the moment
        device = self._model.lm_head.weight.device
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for keyword in keywords:
            tokens.append(self._tokenizer(keyword, return_attention_mask=False)["input_ids"][0])

        while len(tokens) < 4:
            tokens.append(self._special_token)

        encoded_prompt = self._tokenizer.encode(X)

        input_ids = tokens + encoded_prompt
        input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(device)

        greedy_output = self._model.generate(
            input_ids,
            max_length=max_length,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True)

        return self._tokenizer.decode(greedy_output[0]) #, skip_special_tokens=True

    def save(self):
        if self._model is not None and self._trainer is not None:
            print('saving')
            self._trainer.save_model(self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")
    
    def get_token(self, word):
        return self._tokenizer(word, return_attention_mask=False)["input_ids"][0]

    def load(self):
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") # use_fast=True
        self._special_word = '<|bed|>'
        special_tokens_dict = {'additional_special_tokens': [self._special_word]}
        self._tokenizer.add_special_tokens(special_tokens_dict)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        self._special_token = self.get_token(self._special_word)
        print(f"Special token has the number: {self._special_token}. Hopefully this does not vary!")

        if os.path.exists(self._model_path):
            print('loading')
            self._model = GPT2LMHeadModel.from_pretrained(self._model_path, pad_token_id=self._tokenizer.eos_token_id)

        return self

if __name__ == "__main__":
    # execute file with python -m models.GPT2Tuned
    parameters = {"mask_nouns": True, "labels_same_as_input": False, "only_grimm": True, 
    "cos_similarity": True, "add_space": True}
    model = GPT2Keywords(model_id="YY1", parameters=parameters ,server="http://ec2-18-193-73-211.eu-central-1.compute.amazonaws.com:8000")
    model.train()
    # model.predict("Once upon a time there was a horse in the mountains")
    # model.save()
    # model.save_to_server(save_files=True)
