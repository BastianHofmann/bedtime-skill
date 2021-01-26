
import re
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import spacy
import spacy.cli
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from torch.cuda.amp import autocast


class KeywordTrainer(Trainer):

    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        mask_nouns: bool = True,
        filter_loss: list = ["NOUN", 'PROPN'],
        special_token: int = 50257,
        keyword_loss_weight = 1e-4,
        cos_similarity = False,
        ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init ,compute_metrics, callbacks, optimizers)
        self._normal_word = re.compile('^[a-zA-Z\s]+$')
        self._mask_nouns = mask_nouns
        self._special_token = special_token
        self._filter = filter_loss
        self._keyword_loss_weight = keyword_loss_weight
        self._cos_similarity = cos_similarity
        if mask_nouns:
            # switch to gpu
            spacy.prefer_gpu()
            # if package not available download it
            if not spacy.util.is_package('en_core_web_sm'):
                spacy.cli.download('en_core_web_sm')
            self._nlp = spacy.load('en_core_web_sm')
        if cos_similarity:
            self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

    def keyword_forword_step(self, model, inputs):
        """Custom way to compute the loss

        :param model: the pytorch model
        :type model: model
        :param inputs: ['input_ids', 'attention_mask', 'labels', 'keywords']
        :type inputs: dict
        """
        outputs = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'], labels = inputs['labels'])

        with torch.no_grad():
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            embeddings = model.get_input_embeddings()

            # get top 1 prediction for loss compuation
            top1 = torch.argmax(outputs["logits"], dim=2)
            loss_keyword = torch.tensor(0.)
            loss_keyword = loss_keyword.to(device)

            size_keywords = inputs['keywords'].size()
            for batch in range(size_keywords[0]):

                # only create mask if specified
                if self._mask_nouns:
                    # create mask
                    mask_nouns = self.create_noun_mask(top1[batch])
                    # move mask to gpu
                    mask_nouns_gpu = mask_nouns.to(device)

                for index in range(size_keywords[1]):
                    # calc difference between keyword and the output
                    # To do use embeding for comprahension
                    # print(type(masked))
                    if inputs['keywords'][batch][index] == self._special_token:
                        break

                    keyword_em = embeddings(inputs['keywords'][batch][index])
                    pred_em = embeddings(top1[batch])
                    if self._cos_similarity:
                        # add a dimension to the beginning
                        keyword_em.unsqueeze_(0)
                         # 1 means identical
                        difference = 1 - self._cos(pred_em, keyword_em)
                    else:
                        difference = (mask_em - pred_em) ** 2

                    # only mask if specified
                    if self._mask_nouns:
                        mask_difference = difference * mask_nouse_gpu
                    else:
                        mask_difference = difference

                    loss_keyword += mask_difference.sum()

            print(outputs["loss"])
            print(loss_keyword * self._keyword_loss_weight)

        # add exception for testig
        # raise Exception()
        # Save past state if it exists

        outputs["loss"] += loss_keyword * self._keyword_loss_weight #if isinstance(outputs, dict) else outputs[0]
        return outputs

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        outputs = self.keyword_forword_step(model, inputs)

         # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            print("Some past value was set")
            self._past = outputs[self.args.past_index]

        print(outputs["loss"])
        return outputs["loss"]      #if isinstance(outputs, dict) else outputs[0]

    def create_noun_mask(self, batch):
        """Create the mask for the batch

        :param batch: List of tokens
        :type batch: torch.tensor
        :return: mask tensor
        :rtype: torch.tensor
        """
        number_of_samples = batch.size()[0]

        mask_nouns = torch.zeros(number_of_samples, 1, dtype=torch.int8)
        # ignore where pad is
        for i in range(4, number_of_samples):
            word = self.tokenizer.decode(batch[i])

            # skip if not normal word
            if not self._normal_word.match(word):
                continue

            spacy_token = self._nlp(word)


            if not spacy_token:
                print(f"'{word}': {batch[i]}")

            # ignore if stop word
            # add 1 if matches to filter (NOUN, PROPN)
            if not spacy_token[0].is_stop and spacy_token[0].pos_ in self._filter:
                mask_nouns[i] = [1]
        return mask_nouns

    # overide so that loss is correctly calculate
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = self.keyword_forword_step(model, inputs) # Changed this line
            else:
                outputs = self.keyword_forword_step(model, inputs) # Changed this line
            if has_labels:
                if isinstance(outputs, dict):
                    loss = outputs["loss"].mean().detach()
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    loss = outputs[0].mean().detach()
                    logits = outputs[1:]
            else:
                loss = None
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)
