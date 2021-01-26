import functools
import json
# from abc import ABC, abstractmethod
from pathlib import Path
import requests

import numpy as np
from tqdm import tqdm

from .config import ML_DATA_PATH


class Model:
    def __init__(self, model_id: str = None, server: str = None, parameters: dict = None):
        self._tokenizer = None
        self._model = None
        self._model_id = model_id
        self._model_path = ML_DATA_PATH / model_id
        self._server = server
        # will be saved with the model in the db
        self._parameters = parameters
        if not self._model_id:
            self._model_id = self._get_new_model_id()
        # load model
        self.load()

    # @abstractmethod
    def train(self):
        pass

    # @abstractmethod
    def predict(self, X: str) -> np.ndarray:
        pass

    # @abstractmethod
    def save(self):
        pass

    # @abstractmethod
    def load(self):
        pass

    def _check_server_exists(func):
        # https://realpython.com/primer-on-python-decorators/
        @functools.wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            # check if server was specified
            if not self._server:
                raise Exception("No server was specified")
            # TODO test if server can be pinged
            return func(self, *args, **kwargs)
        return wrapper_decorator

    @_check_server_exists
    def save_to_server(self, save_files=True):
        """
        Upload model files to server and save parameters to db
        """
        url = self._server + "/addmodel"
        class_name = self.__class__.__name__
        parameters = json.dumps(self._parameters)
        request = {"model_id": self._model_id, "class_name": class_name, "parameters": parameters}
        r = requests.post(url, json=request)

        if r.status_code != 200:
            print("The Model could not been uploaded")
            return False
        else:
            respond = r.json()
            if respond["success"]:
                print(f"The model with the id {self._model_id} was successfully added to the database")
                print(f"Now uploading files")
                if save_files:
                    return self._upload_files()
        return True

    def _upload_files(self):

        # to implement it with procress bar the package requests_toolbelt is
        # required with is not installed in colab
        url = self._server + "/uploadmodel"
        request = [('model_id', (None, self._model_id))]
        for extension in ["*.json", "*.h5", "*.bin"]:
            for current_file in self._model_path.glob(extension):
                request.append(('files', (current_file.name, open(current_file, 'rb'))))
        r = requests.post(url, files=request)
        if r.status_code != 200:
            print("The Model data could not been uploaded")
            return False
        else:
            return True

    @_check_server_exists
    def upload_prediction(self, keyword1: str, text: str, keyword2: str = None, keyword3: str = None) -> bool:
        """Upload a prediction to the server
        """
        url = self._server + "/addprediction"
        if not keyword1 or not text:
            print("Keyword1 and text is required")
            return False
        request = {"model_id": self._model_id, "keyword1": keyword1,
        "text": text, "keyword2": keyword2, "keyword3": keyword3}
        r = requests.post(url, json=request)

        if r.status_code != 200:
            print("The prediction could not be uploaded")
            return False
        else:
            respond = r.json()
            return respond["success"]

    @_check_server_exists
    def _get_new_model_id(self) -> str:
        """Get unique id form server
        """
        url = self._server + "/getuniqueid"
        r = requests.post(url)

        if r.status_code != 200:
            print("A unique id could not be created")
            return None
        else:
            respond = r.json()
            return respond["data"]
