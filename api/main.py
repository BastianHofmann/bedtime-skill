import shutil
from os.path import splitext
from typing import List
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Json, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_404_NOT_FOUND
from starlette.responses import FileResponse

from .ml.models.config import ML_DATA_PATH
from .db.db_handler import DBHandler
# from .ml.GPT2 import get_model
# from .ml.model import Model


class PredictRequest(BaseModel):
    data: str

class AddPredictionRequest(BaseModel):
    model_id: str
    keyword1: str
    text: str
    keyword2: str = None # default is None
    keyword3: str = None

    @validator('model_id')
    def id_three_char(cls, v):
        assert len(v) == 3, 'must be three chars long'
        assert v.isalnum(), 'must be alphanumeric'
        assert v.isupper(), 'must be upper case letters only'
        db = DBHandler()
        assert db.id_exist(v), 'model_id does not exists'
        return v

class AddModelRequest(BaseModel):
    model_id: str
    class_name: str
    parameters: Json = None # default is None

    @validator('model_id')
    def id_three_char(cls, v):
        assert len(v) == 3, 'must be three chars long'
        assert v.isalnum(), 'must be alphanumeric'
        assert v.isupper(), 'must be upper case letters only'
        db = DBHandler()
        assert not db.id_exist(v), 'model_id already exists'
        return v

class ModelRequest(BaseModel):
    model_id: str

    @validator('model_id')
    def id_three_char(cls, v):
        assert len(v) == 3, 'must be three chars long'
        assert v.isalnum(), 'must be alphanumeric'
        assert v.isupper(), 'must be upper case letters only'
        db = DBHandler()
        assert db.id_exist(v), 'model_id does not exists'
        return v

class DownloadRequest(BaseModel):
    model_id: str
    filename: str

    @validator('model_id')
    def id_three_char(cls, v):
        assert len(v) == 3, 'must be three chars long'
        assert v.isalnum(), 'must be alphanumeric'
        assert v.isupper(), 'must be upper case letters only'
        db = DBHandler()
        assert db.id_exist(v), 'model_id does not exists'
        return v


class GetPredictionRequest(BaseModel):
    keyword1: str
    keyword2: str = None # default is None
    keyword3: str = None


class PredictResponse(BaseModel):
    data: str

class SuccessResponse(BaseModel):
    success: bool

class IdResponse(BaseModel):
    data: str

class ModelResponse(BaseModel):
    class_name: str
    creation_timestamp: str
    parameters: Json
    file_names: List[str]


app = FastAPI()


# @app.post("/predict", response_model=PredictResponse)
# def predict(input: PredictRequest, model: Model = Depends(get_model)):
#     X = input.data
#     y_pred = model.predict(X)
#     result = PredictResponse(data=y_pred)

#     return result

@app.post("/addprediction", response_model=SuccessResponse)
def addprediction(input: AddPredictionRequest):
    db = DBHandler()
    success = db.add_prediction(input.model_id, input.keyword1, input.text, input.keyword2, input.keyword3)
    return {"success": success}

@app.post("/getprediction")
def getprediction(input: GetPredictionRequest):
    db = DBHandler()
    response = db.get_prediction(input.keyword1, input.keyword2, input.keyword3)
    return response

@app.post("/getuniqueid", response_model=IdResponse)
def getuniqueid():
    db = DBHandler()
    response = db.create_unique_id()
    return {"data": response}

@app.post("/addmodel", response_model=SuccessResponse)
def addmodel(input: AddModelRequest):
    db = DBHandler()
    success = db.add_model(input.model_id,input.class_name, input.parameters)
    return {"success": success}

@app.post("/uploadmodel")
def uploadmodel(files: List[UploadFile] = File(...), model_id: str = Form(...)):
    path = ML_DATA_PATH / model_id
    # create folder for
    path.mkdir(parents=True, exist_ok=True)

    for data_file in files:
        # check if the file has a correct extension
        check_extension(data_file.filename)

        filepath = path / data_file.filename
        # save the uploaded file
        try:
            with filepath.open("wb") as buffer:
                shutil.copyfileobj(data_file.file, buffer)
        finally:
            data_file.file.close()

@app.post("/getmodel")
def getmodel(input: ModelRequest):
    # path to the model files
    path = ML_DATA_PATH / input.model_id

    db = DBHandler()
    result = {}
    result["file_names"] = get_file_names(path)
    result["class_name"], result["creation_timestamp"], result["parameters"] = db.get_model(input.model_id)
    return result

def get_file_names(path: Path):
    """Return the files in a dic with have the extension json or h5

    :param path: file path
    :type path: Path
    :return: Array of files names
    :rtype: list[str]
    """
    names = []
    for extension in ["*.json", "*.h5", "*.bin"]:
        for currentFile in path.glob(extension):
            names.append(currentFile.name)
    return names

def check_extension(filename):
    """Checks if the extension of the file is json or h5

    :param filename: filename to check
    :type filename: str or Path
    :raises HTTPException: Otherwise raise 422 error
    """
    extension = splitext(filename)[1]
    if extension != ".json" and extension != ".h5" and extension!= ".bin":
        raise HTTPException(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"The uploaded files do not match the conditions",
        )

@app.post("/downloadmodelfile")
def downloadmodelfile(input: DownloadRequest):
    path = ML_DATA_PATH / input.model_id

    # check if extension if ok
    check_extension(input.filename)


    filepath = path / input.filename

    # check if file exists
    if filepath.is_file():
        return FileResponse(filepath)
    else:
        raise HTTPException(
        status_code=HTTP_404_NOT_FOUND,
        detail=f"The requested file could not been found on the server",
        )



