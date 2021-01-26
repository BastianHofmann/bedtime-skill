import importlib
import json
from pathlib import Path

import requests
from tqdm import tqdm

from models.config import ML_DATA_PATH
from models.model import Model


class ModelLoader:

    def __init__(self, model_id: str, server: str):
        if len(model_id) != 3:
            raise Exception("Model id must have 3 characters")
        if not server:
            raise Exception("Server must be defined in order to download model")
        self._model_id = model_id
        self._server = server
        self._model_path = ML_DATA_PATH / model_id

    def get_model(self) -> Model:
        """Get the model which fits to the id and is based on the abstract class model

        :raises Exception: Server not available
        :return: The model class
        :rtype: Model
        """
        respond = self._download_model()
        if not respond:
           raise Exception(f"Model does not exist with id: {self._model_id} or the server {self._server} does not respond") 

        # create model weights folder if it does not exist
        self._model_path.mkdir(parents=bool, exist_ok=True)
        # download files with are not available
        for filename in respond["file_names"]:
            file = self._model_path / filename
            if not file.is_file():
                self._download_file(filename)

        print(f'Trying to load the model: {respond["class_name"]} with was created on the {respond["creation_timestamp"] }')
        return self._import_model_class(respond["class_name"], respond["parameters"])
    
    def _download_model(self):
        """Try to download model from server
        """
        # create dir if not already exists
        self._model_path.mkdir(parents=True, exist_ok=True)

        url = self._server + "/getmodel"
        request = {"model_id": self._model_id}
        r = requests.post(url, json=request)

        if r.status_code != 200:
            print("The Model config could not been downloaded")
            return False
        else:
            respond = r.json()
            # class_name = respond["class_name"]
            # creation_timestamp = respond["creation_timestamp"] 
            # if not none then convert json to dict
            if respond["parameters"]:
                respond["parameters"] = json.loads(respond["parameters"])
            return respond

    def _import_model_class(self, class_name: str, parameters: dict) -> Model:
        """Dynamically imports the correct class for the model

        :param class_name: The file name and class name (must be the same)
        :type class_name: str
        :param parameters: The parameters saved on the server
        :type parameters: dict
        :raises Exception: If file not exits raise an error
        :return: returns the model object
        :rtype: Model
        """
        class_file = Path(__file__).resolve().parent / "models" / f"{class_name}.py"
        if not class_file.is_file():
            raise Exception("Fatal error: class from db does not exist")
        test = importlib.import_module(f'.{class_name}', '.models')
        model_class = getattr(test, class_name)
        model = model_class(model_id=self._model_id, server=self._server, parameters=parameters)
        return model

    def _download_file(self, filename: str):
        """Downloads the files for the model in the weights folder

        :param filenames: file with have to been downloaded
        :type filenames: list[str]
        """
        file = self._model_path / filename
        url = self._server + "/downloadmodelfile"
        request = {"model_id": self._model_id, "filename": filename}
        
        r = requests.post(url, stream=True, allow_redirects=True, json=request)
        total_size = int(r.headers.get('content-length'))
        initial_pos = 0
        with open(file,'wb') as f: 
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename,initial=initial_pos) as pbar:
                for ch in r.iter_content(chunk_size=1024):                             
                    if ch:
                        f.write(ch) 
                        pbar.update(len(ch))


if __name__ == "__main__":
    # class_name = "GPT2"
    # test = importlib.import_module(f'.{class_name}', '.models')
    # model_class = getattr(test, class_name)
    # model = model_class(model_id="TX3")
    # model_loader = ModelLoader("TX3", "http://127.0.0.1:8000")
    model_loader = ModelLoader("NY3", "http://ec2-18-193-73-211.eu-central-1.compute.amazonaws.com:8000")
    model = model_loader.get_model()
    print(model.predict("The man went skiing "))
    