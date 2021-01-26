# Bedtime Stories

This repositories provides tools for bedtime story generation using two
different kinds of GPT-2 models.

The models are hosted using fast API and can even be loaded locally and saved
back to the server. Pre-saved predictions can also be accessed through the API.

## Project setup
1. Create the virtual environment.
```
virtualenv /path/to/venv --python=/path/to/python3
```
You can find out the path to your `python3` interpreter with the command `which python3`.

2. Activate the environment and install dependencies.
```
source /path/to/venv/bin/activate
pip install -r requirements.txt
```

3. Launch the service
```
uvicorn api.main:app
```

## Posting requests locally
When the service is running, try
```
127.0.0.1/docs
```
or 
```
curl
```

## Deployment with Docker
1. Build the Docker image
```
docker build --file Dockerfile --tag bedtime-stories .
```

2. Running the Docker image
```
docker run -p 8000:8000 bedtime-stories
```

3. Entering into the Docker image
```
docker run -it --entrypoint /bin/bash bedtime-stories
```

## docker-compose
1. Launching the service
```
docker-compose up
```
This command looks for the `docker-compose.yaml` configuration file. If you want to use another configuration file,
it can be specified with the `-f` switch. For example  

2. Testing
```
docker-compose -f docker-compose.test.yaml up --abort-on-container-exit --exit-code-from bedtime-stories
```

3. Remove data
```
docker-compose down -v
```

4. Connect to mysql instance in docker container
```
docker run -it --entrypoint #id /bin/bash
mysql -u bedtime -p
```

# Model Loader Usage

To load a model locally you can use the model loader like this:
```python
from model_loader import ModelLoader

model_loader = ModelLoader("XYZ", "http://localhost:8000")
model = model_loader.get_model()
```

You can now use the model like it was trained locally:
```python
pred = model.predict("Once upon a time there was a boat and a dog", keywords=['boat', 'dog'], max_length=300)
```
