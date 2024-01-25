# Machine Learning Engineering Course

A two-day course designed to equip participants with practical skills in Machine Learning (ML) Engineering, transitioning from data science prototypes to production-level ML solutions.


**Day 1: Training Pipeline** 

Transforming a Jupyter notebook into a training job.

* Turning the notebook into an executable python file
* Best practices: folder structure, documenting, logging, testing

Outcome: A parametrized, reproducible ML training pipeline.

**Day 2: Deployment**

Deploying the trained ML model in a web service.

* Flask
* Docker
* Deployment to the cloud

Outcome: A deployable web service hosting the ML model.

# Day 1: Training Pipeline

Notebook --> training pipeline

* Environment setup
  * Codespaces
* Project structure
* Virtual environments
* Transforming a notebook into a script
  * Refactoring the script
  * Parametrizing the script
* Best practices:
  * Readme
  * Documentation
  * Logging
  * Modularization
  * Testing
  * Makefile



## Problem

Taxi trip duration prediction

- Dataset: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Goal: predict the time to travel from point A to point B
- Notebook: https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/blob/main/01-train/duration-prediction-starter.ipynb

## Environment Setup

Overview:

- GitHub Codespaces
- Installing the required packages
- Alternative setups

Note: You don't have to use GitHub codespaces, you can do it locally

### Creating a github repo

Create a repository (e.g. "mle-workshop-day1-train")

### Creating a codespace

![image](https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/assets/875246/15529150-74a5-4295-9f5c-a9de857b6ac6)


### Opening in VS Code

- File -> Open in VS Code Desktop
- When asked, allow installing the extension and authenticate with GitHub
- You have an enrironment ready!
- Test that docker works: `docker run hello-world`

### Installation 

This is the code part - you will need to run it regardless on the environment you use

- Docker (already installed in Codespaces)
- `pip install pipenv`
- `python -V` to check python version (use miniconda or devcontainers if you need a different version)

Links:

- [Setting up docker and python on Ubuntu](https://github.com/alexeygrigorev/hands-on-mlops-workshop?tab=readme-ov-file#installing-all-the-necessary-software-and-libraries) 

### Stopping the codespace

![image](https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/assets/875246/760c27a7-6dc7-4486-a0fd-c0169e81ca60)

- Codespaces -> Stop current codespace
- If you need to keep it: codespaces -> select the code space -> untick "auto-delete codespace"


Links:

- [Codespaces overview](https://docs.github.com/en/codespaces/overview)
- [Billing](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces)
- Further learning: [Python Dev Containers](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/setting-up-your-python-project-for-codespaces)



## Project structure

Overview:

- Best practices in structuring ML training pipelines
- Cookiecutter Data Science


### Basic folder structure

We need a separate folder for:

- `notebooks`: notebooks for keeping experimental notebooks and one-off reports
- `duration_prediction`: source code for the training pipeline (the name depends on your project)
- `tests`: folder for tests
- `data`: folder for data, but in practice we often use external data stores like s3/dvc (out of the scope)
- `models`: folder for models, but in practice we often use model registries like mlflow (out of the scope)


So we will create the following folders:

```bash
mkdir duration_prediction
mkdir tests
mkdir notebooks
mkdir data
mkdir models
touch README.md
touch Makefile
```

### Cockiecutter (optional)

For more advanced folder structure, check cookiecutter data science. It creates a [folder/file structure](https://github.com/drivendata/cookiecutter-data-science?tab=readme-ov-file#the-resulting-directory-structure) with:

- Folders for data and models 
- Folders for notebooks and scripts
- Makefile

Installing it

```bash
pip install cookiecutter
```

(Installed globally with `pip` rather than individually for each project with `pipenv`)

Creating the repository:

```bash
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```


## Virtual environment

Overview:

- Virtual environments and virtual environment managers
- Pipenv and installing pipenv
- Preparing the environment 


### Why needed

Virtual environments

- Dependency management: manages project dependencies
- Isolation: separates the project dependencies from others projects
- Replicability: helps with environment duplication across machines

Virtual environment managers

- Pipenv, conda, poetry
- Simplifies environment (venv) and dependency management (pip)
- Easier than pip+virtualenv separately

### Pipenv

- pip+venv in one tool
- Creates and manages environments per project
- `Pipfile.lock`
- See [here](https://pipenv.pypa.io/en/latest/) for more information about pipenv 

Install pipenv:

```bash
pip install pipenv
```

Create the enviroment

```bash
pipenv install
```

If you want to create an environment with a specific python version, you can use the `--python` flag, e.g.

```bash
pipenv install --python=3.10
```

If your environment doesn't have the version of Python you need, there are multiple options:

- `pyenv` - works well with pipenv
- `conda` - for getting the python binary only, the env is managed by pipenv (works better on Windows)   


With `conda`, it will looks like that:

```bash
conda create --name py3-10-10 python=3.10.10
# on windows
pipenv install --python=/c/Users/alexe/anaconda3/envs/py3-10-10/python.exe
# or on linux
pipenv install --python=~/miniconda3/envs/py3-10-10/bin/python
```

Let's look at `Pipfile`

### Preparing the project environment

Install the packages:

```bash
pipenv install scikit-learn==1.2.2 pandas pyarrow

pipenv install --dev jupyter

# if need visualizations
pipenv install --dev seaborn
```

Dev dependencies: only for local environment, not deployed to production

Let's look again at `Pipfile` and also at `Pipfile.lock`

Note: on Linux you might also need to instal `pexpect` for jupyter:

```bash
pipenv install --dev jupyter pexpect
```


## Notebook to Script Transformation

Overview: 

- Converting and parameterizing a Jupyter notebook into a Python script
- Job parametrization
- Saving the model for deployment


### Running the notebook

- We'll start with the model [we already created previously](01-train/duration-prediction-starter.ipynb)
- Copy this notebook to "duration-prediction.ipynb"
- This model is used for preducting the duration of a taxi trip

```bash
cd notebooks
wget https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/raw/main/01-train/duration-prediction-starter.ipynb
mv duration-prediction-starter.ipynb duration-prediction.ipynb
```

In this notebook, We use the data from the [NYC TLC website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

- Train: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
- Validation: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet


Run jupyter

```bash
pipenv run jupyter notebook
```

Forward the 8888 port if you're running the code remotely (In VS Code: Ports -> Forward a Port)

Now open http://localhost:8888/


Adding a scikit-learn pipeline

- Now two artifacts (dictionary vectorizer and the model), more difficult to manage
- Let's convert them into a single pipeline

```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LinearRegression(**model_params)
)

pipeline.fit(train_dicts, y_train)
y_pred = pipeline.predict(val_dicts)
```

And save it with pickle


### Converting the notebook into a script

To convert the notebook to a python script, run

```bash
pipenv run jupyter nbconvert --to=script duration-prediction.ipynb
```

Rename the file to `train.py` and clean it

- Move all the imports to the top
- Put the training logic into one function: train
- Create a parametrized `run` function

Run it:

```bash 
pipenv run python train.py
```

### CLI

- Now the values are hardcoded
- We want to set them when executing the script, for example, with command line parameters


```bash 
pipenv run python train.py \
    --train-month=2022-01 \
    --validation-month=2022-02 \
    --model-output-path=../models/model-2022-01.bin
```

There's a built-in module `argparse` that we can use for that:

```python
import argparse
from datetime import date

def run(train_date, val_date, model_output_path):
    # Your existing code to train the model
    pass

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model based on specified dates and save it to a given path.')
parser.add_argument('--train-month', required=True, help='Training month in YYYY-MM format')
parser.add_argument('--validation-month', required=True, help='Validation month in YYYY-MM format')
parser.add_argument('--model-save-path', required=True, help='Path where the trained model will be saved')

args = parser.parse_args()

# Extract year and month from the arguments and convert them to integers
train_year, train_month = args.train_month.split('-')
train_year = int(train_year)
train_month = int(train_month)

val_year, val_month = args.validation_month.split('-')
val_year = int(val_year)
val_month = int(val_month)

# Create date objects
train_date = date(year=train_year, month=train_month, day=1)
val_date = date(year=val_year, month=val_month, day=1)

# Call the run function with the parsed arguments
model_output_path = args.model_output_path
run(train_date, val_date, model_save_path)
```

There are also packages like [click](https://click.palletsprojects.com/en/8.1.x/)
that make it even simpler.

First, install click:

```bash
pipenv install click
```

```python
import click
from datetime import date

def run(train_date, val_date, model_output_path):
    # Your existing code to train the model
    pass

@click.command()
@click.option('--train-month', required=True, help='Training month in YYYY-MM format')
@click.option('--validation-month', required=True, help='Validation month in YYYY-MM format')
@click.option('--model-output-path', required=True, help='Path where the trained model will be saved')
def main(train_month, validation_month, model_output_path):
    train_year, train_month = args.train_month.split('-')
    train_year = int(train_year)
    train_month = int(train_month)

    val_year, val_month = args.validation_month.split('-')
    val_year = int(val_year)
    val_month = int(val_month)

    train_date = date(year=train_year, month=train_month, day=1)
    val_date = date(year=val_year, month=val_month, day=1)

    run(train_date, val_date, model_output_path)


if __name__ == '__main__':
    main()
```


## Best practices

Outline

- Readme
- Documentation
- Logging
- Modularization
- Testing
- Makefile

### README.md

This is the most important file in your project 

- First Impression: it's the first document users see
- Project description: it explains what the project is about, its purpose, and its functionality
- Guidance: it helps new users understand how to get started, install, and use the project
- Documentation reference: provides links to more detailed documentation (if available)
- Contribution guidelines: describes the involvement process if the project is open for collaboration

What a README.md of a portfolio project can contain:

- [ ] **Project title**
- [ ] **Brief description**: A short summary explaining what the project does
- [ ] **Installation instructions**: step-by-step guide on how to set up and run the project
- [ ] **Dependencies**: any libraries, frameworks, or tools that the project needs (outside of requirements.txt/pipenv). It includes: s3 buckets, databases, special hardware, 
- [ ] **Configuration**: details on configuring the project
- [ ] **Structure**: files and folders in the project 
- [ ] **Usage**: examples of how to run the project and its parts locally, including code snippets and command line, running tests, running it with docker, etc
- [ ] **Deployment**: where and how the project is deployed and how to update it
- [ ] **Access**: if the project is deployed, how to access it and how to use it
- [ ] **Documentation**: extenal more detailed documentation or helpful information (articles, wikis, ect, if applicable)
- [ ] **Contributing**: guidelines for project contributions (if applicable)
- [ ] **License**: The license under which the project is released (if applicable)
- [ ] **Acknowledgments**: Credits to any third-party resources or contributors (if applicable)

The order is arbitrary, you can follow any other order

See a small example [here](01-train/README.md)


Other examples:

- https://github.com/Isaac-Ndirangu-Muturi-749/Kitchenware-Image-Classification-System-with-Keras-Deploying-and-Testing-with-AWS-Lambda
- https://github.com/cyberholics/Malicious-URL-detector
- https://github.com/AndrewTsai0406/Legal-AI-Judge
- https://github.com/mroubaud/mlz-capstone-project-1
- https://github.com/lumenalux/mlzoomcamp-2023-capstone-1
- https://github.com/abhirup-ghosh/facial-expression-classifier-app

Note that these are examples from students. They are not ideal, but should give you some ideas how a README can look like


### Documentation

Why do we need docstrings:

* We work in a team, not alone, so we need to have good description of how functions/classes works
* Docstrings integrate with documentation tools to automatically generate documentation
* Especially important if we work on a library

Tip: use ChatGPT for that. Example prompt: "create a docstring for the following function"

```python
def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a Parquet file into a DataFrame and preprocess the data.

    This function reads a Parquet file, computes the duration of each taxi trip,
    filters the trips based on duration criteria, and converts certain categorical
    columns to string type.

    Parameters:
    filename (str): The path to the Parquet file.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Function implementation...
    pass


def train(train_month: datetime, val_month: datetime, model_output_path: str) -> None:
    """
    Train a linear regression model for predicting taxi trip durations.

    This function trains a model using data from specified months, evaluates it,
    and saves the trained model to a file. It reads data, preprocesses it,
    fits a linear regression model within a pipeline, and evaluates the model
    using RMSE. The trained model is then saved to the specified path.

    Parameters:
    train_month (datetime): The month for training data.
    val_month (datetime): The month for validation data.
    model_output_path (str): The file path to save the trained model.

    Returns:
    None
    """
    pass
```

### Logging

We often include useful information with `print` statements - it helps us understand what's happening in the code and find bugs.


Why can't we just use print statements?

- Not flexible: can't turn them on and off as we need based on their importance (all of them are equally important)
- Limited information: they don't provide enough information like timestamps or source
- No structure: often no consistent format, which makes it harder to analyze logs later

Logs allow:

* Categorizing Issues: differentiate messages by severity levels (info, debug, error)
* Timestamping Events: provide exact times for each event, crucial for troubleshooting
* Controlling Output: toggle or filter log messages based on the environment (development, production) or other criteria


Let's use logging in our script:

```python
import logging

logger = logging.getLogger(__name__)

def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Docstring here...
    """
    logger.info(f"Reading data from {filename}")

    try:
        df = pd.read_parquet(filename)
        # Rest of your code...
        logger.debug(f"Dataframe shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise

def train(train_month: datetime, val_month: datetime, model_output_path: str) -> None:
    """
    Docstring here...
    """
    logger.info(f"Training model for {train_month} and validating for {val_month}")
    try:
        # ...
        logger.debug(f"URL for training data: {url_train}")
        logger.debug(f"URL for validation data: {url_val}")

        # Your code to train the model...
        logger.info(f'RMSE: {rmse}')
        # ...
        logger.info(f"Model trained successfully. Saving model to {model_output_path}")
        # Save model...
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise
```

Configure logging


```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```
We can set the level to `DEBUG` to also see the debug messages


### Modularization

We have all the code in one file. Let's split it into two:

- `main.py` for running the whole thing
- `train.py` for logic for training

Also, we need to create an empty `__init__.py` file, otherwise `duration_prediciton` won't be recognized as a python module

Why modules?

* Easier to maintain and update small files
* Reusable - can use the module in some other place 
* Useful for testing - we will see later


Now run it like that:

```bash
TRAIN="2022-01"
VAL="2022-02"

pipenv run python \
    -m duration_prediction.main \
    --train-month="${TRAIN}" \
    --val-month="${VAL}" \
    --model-output-path="./models/model-${TRAIN}.bin"
```


### Testing

Why testing:

* Error Detection: find bugs and errors before deployment
* Quality Assurance: make sure that the behavior is expected (follows specifications - if applicable)
* Refactoring: allows us making changes to the code without worrying that we accidentally break something
* Documentation: it's also a form of documentation, showing how the code is intended to be used

We will spend more time talking about testing tomorrow, but for now we can create a simple test.

Let's create a file `test_train.py` in `tests`:

```python
import os
import unittest
from datetime import datetime

from duration_prediction.train import train


class TestMLPipeline(unittest.TestCase):

    def test_train_function(self):
        # Test if train function works and saves the model file
        
        train_month = datetime(2022, 1, 1)
        val_month = datetime(2022, 2, 1)
        model_output_path = 'test_model.bin'
        
        train(train_month, val_month, model_output_path)

        # Check if model file is created
        self.assertTrue(os.path.exists(model_output_path))

        # Remove test model file after test
        os.remove(model_output_path)


if __name__ == '__main__':
    unittest.main()
```

Run only one test:

```bash
pipenv run python -m unittest tests.test_train
```

Running all tests in `tests/`

```bash
pipenv run python -m unittest discover -s tests
```

Enabling tests in VS code

* You can also run tests in VS Code
* Install the Python extension for that
* Select the right interpreter: Ctrl+Shift+P -> Python -> Select interpreter
* Install pytest as a dev dependency `pipenv install --dev pytest`


### Makefile

* To run the unit tests, we need to use a long command.
* Instead, we can just do `make tests`

```makefile
tests:
	pipenv run python -m unittest discover -s tests
```

You might get something like that:
```bash
$ make tests
make: 'tests' is up to date.
```

Fix:
```makefile
.PHONY: tests
tests: 
	pipenv run python -m unittest discover -s tests
```

It's a very useful tool for automation and creating shortcuts. We will talk about it more on the next day



## Homework

- Refactor your own code
- You can use [this notebook](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/code/05-train-churn-model.ipynb) (this is the [data](https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv) for the notebook)


# Day 2: Deployment

Deploying the trained ML model in a web service.

Outcome: A deployable web service hosting the ML model.


## Serving with Flask

Outline:

- Preparing the env
- Basics of Flask
- Serving a model with Flask


### Preparing the environment

Like yesterday, let's create a repo on github ("mle-workshop-day2-deploy") and create a codespace there (or use your local/other env)


Prepare the environment

- `pip install pipenv` to install pipenv
- `pipenv install` to create the env
- `pipenv install flask scikit-learn==1.2.2`


Folder structure

```
mkdir models
mkdir duration_prediction_serve
mkdir tests
mkdir integration_tests
touch README.md
touch Makefile
touch Dockerfile
```

### Flask 101

Flask is a web server. Let's create the simplest flask app:

```python
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)
```

Run it:

```bash
pipenv run python ping.py
```

Open in the browser: http://localhost:9696/ping

Or query with curl:

```bash
curl http://localhost:9696/ping
```

### Serving the model

First, let's download the model:

```bash
wget https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop/raw/main/01-train/models/model-2022-01.bin -P models
```

Now let's load it and process a request (`serve.py`):

```python
import pickle

with open('./models/model-2022-01.bin', 'rb') as f_in:
    model = pickle.load(f_in)

trip = {
    'PULocationID': '43',
    'DOLocationID': '238',
    'trip_distance': 1.16
}

prediction = model.predict(trip)
print(prediction[0])
```

We can now combine the two scripts into one:

```python
import pickle

from flask import Flask, request, jsonify

with open('./models/model-2022-01.bin', 'rb') as f_in:
    model = pickle.load(f_in)


# "feature engineering"
def prepare_features(ride):
    features = {}
    features['PULocationID'] = str(ride['PULocationID'])
    features['DOLocationID'] = str(ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features



def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)
    # if we had post-processing it'd go here
    # result = post_process(pred)

    result = {
        'preduction': {
            'duration': pred,
        },
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

Run it:

```bath
pipenv run python duration_prediction_serve/serve.py
```

We can add it to the makefile:

```makefile
run:
    pipenv run python duration_prediction_serve/serve.py
```

And do `make run`

Now let's test it:

```bash
REQUEST='{
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}'

URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL}
```

We can write this to `predict-test.py` (note - this is not a python "test", just a script for sending a request and printing it):

```python
import requests

url = 'http://localhost:9696/predict'

trip = {
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}

response = requests.post(url, json=trip).json()
print(response)
```

Install requests:

```bash
pipenv install --dev requests
```

Run it:

```bash
pipenv run python predict-test.py
```


### Configuring with env variables

We have one problem: the model path is hardcoded. How can we make it more flexible?

With environment variables

```bash
export MODEL_PATH="./models/model-2022-01.bin"
```

Loading it:

```python
import os
import pickle

from flask import Flask, request, jsonify

MODEL_PATH = os.getenv('MODEL_PATH', 'model.bin')

with open(MODEL_PATH, 'rb') as f_in:
    model = pickle.load(f_in)
```

In Makefile, if the variable is only specific to one recipe: 

```makefile
run:
	export MODEL_PATH="./models/model-2022-01.bin"; \
	pipenv run python duration_prediction_serve/serve.py
```

If you need it for all tasks:

```makefile
MODEL_PATH = ./models/model-2022-01.bin
export MODEL_PATH

run:
	pipenv run python duration_prediction_serve/serve.py
```

(Important - no quotes for the value, quotes would be passed to the variable)


Often we also want to know which version of the model is served. It's very useful later for debugging.

For that, we can also pass the version to the env variable:

```bash
MODEL_PATH = ./models/model-2022-01.bin
export MODEL_PATH

VERSION = 2022-01-v01
export VERSION

run:
	pipenv run python duration_prediction_serve/serve.py
```

Using it:

```python
VERSION = os.getenv('VERSION', 'N/A')

result = {
    'preduction': {
        'duration': pred,
    },
    'version': VERSION
}
```


## Dockerization and Containerization

Outline

- Docker intro
- Containerizing the web service

### Docker 

With virtual envs we have some isolation, but it's not enough

- Our app can still affest other apps running on the same machine
- There are system dependencies (specific version of unix, unix apps, etc) that our app may have
- We want to have total control over the entire environment and 100% reproducibility

For that we use containers, and Docker allows to containerize our app 

- 100% same env locally and when deployment
- Simple to deploy: test locally and roll out to any container management system (Kubernetes, Cloud Run, ECS, anything else)


### Installing Docker

Installing Docker on Ubuntu (skip for Codespaces)

```bash
sudo apt update
sudo apt install docker.io

# to run docker without sudo
sudo groupadd docker
sudo usermod -aG docker $USER

# now log in/out

docker run hello-world
```

### Running base images

Let's run an image with clean ubuntu and python:

```bash
docker run -it python:3.10
```

- `-it` run in interactive mode, with access to the terminal
- `python` - image name
- `3.10` - image tag

Instead of Python, let's have bash:


```bash
docker run -in --entrypoint=bash python:3.10
```

- Everything you pass before the tag are parameters for docker
- Everything after the tag - CLI parameters for the process in the container

Let's do something stupid (inside the container!!!)

```bash
rm -rf --no-preserve-root /
```

Close the process, open again. Everything is back again. So containers are "stateless" - the state is not preserved

Which is also not always good. Let's say we want to install pandas:

```bash
pip install pandas
```

But when we stop and start the container again, it's gone. 


### Dockerfile

That's why we need a `Dockerfile`, which contains all the setup instructions. It's used to build the image that we later run in the container 

```dockerfile
FROM python:3.10

RUN pip install pandas

ENTRYPOINT [ "bash" ]
```

Let's build this image:

```bash
docker build -t local-pandas:3.10 . 
```

- `-t` for specifying the tag 
- `.` to run in the current directory (it'll see all the files there, including Dockerfile)

And run it:

```bash
docker run -it local-pandas:3.10
```

### Packaging the Service

Now let's create a Dockerfile for our flask app 

```dockerfile
FROM python:3.10

RUN pip install pipenv

ENTRYPOINT [ "bash" ]
```

Build it:

```bash
docker build -t duration-prediction .
```

Run it:

```bash
docker run -it duration-prediction:latest
```

When we don't specify the tag, it's `latest`

To make it simpler, let's add these two recepies to `Makefile`:

```makefile
docker_build:
	docker build -t duration-prediction .

docker_run: docker_build
	docker run -it duration-prediction
```

Now we can just do `make docker_run`


Let's finish the dockerfile. The final result

```dockerfile
FROM python:3.10

RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY duration_prediction_serve duration_prediction_serve
COPY models/model-2022-01.bin model.bin

ENV MODEL_PATH model.bin
ENV VERSION 2022-01-v01

EXPOSE 9696

ENTRYPOINT [ "python", "duration_prediction_serve/serve.py" ]
```

Command to run it:

```makefile
docker_build:
	docker build -t duration-prediction .

docker_run: docker_build
	docker run -it -p 9696:9696 duration-prediction
```

- `-p <container_port>:<host_port>` - maps a port on the container to a port on the host machine (9696 in container -> 9696 on the host)

Verify that it's working:

```bash
python predict-test.py
```

Note the warning:

> WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.

Let's fix it by using "a production WSGI server instead":

```bash
pipenv install gunicorn
```

```dockerfile
ENTRYPOINT [ \
    "gunicorn", \ 
    "--bind=0.0.0.0:9696", \ 
    "duration_prediction_serve.serve:app" \ 
]
```

Re-build it and re-run


## Testing

Outline

- Unit tests
- Integration tests

### Unit tests

- Yesterday we did very simple tests for the pipeline
- Let's make simple tests for the app 

First, we need to split the logic into two files:

- `serve.py` for the main serving logic
- `features.py` for the feature engineering part (in this case it's kind of meaningless but in real project you'd have more logic there)

Test it. We will need to adjust our `run` recipe:

```makefile
pipenv run python -m duration_prediction_serve.serve
```

We will use pytest instead of built-in unittest:

```bash
pipenv install --dev pytest
```

Now let's make a test in `tests/test_features.py`:

```python
import pytest

from duration_prediction_serve.features import prepare_features


def test_prepare_features_with_valid_input():
    ride = {
        'PULocationID': 123,
        'DOLocationID': 456,
        'trip_distance': 7.25
    }

    expected = {
        'PULocationID': '123',
        'DOLocationID': '456',
        'trip_distance': 7.25
    }

    result = prepare_features(ride)

    assert result == expected


def test_prepare_features_with_missing_keys():
    ride = {
        'trip_distance': 3.5
    }

    with pytest.raises(KeyError):
        prepare_features(ride)
```

Run it: 

```bash
pipenv run pytest tests/
```

We might need to add the current directory to `PYTHONPATH` if pytest can't find our module:

```bash
PYTHONPATH=. pipenv run pytest tests/
```

Alternative - create a `pyproject.toml` file with pytest configuration:

```ini
[tool.pytest.ini_options]
pythonpath = ["."]
```

Add a recipe and let `docker_build` depend on it:

```makefile
.PHONY: tests
tests:
	pipenv run python -m unittest discover -s tests

docker_build: tests
	docker build -t duration-prediction .
```

(So we don't build unless tests pass)

Now let's do `docker_run` 


### Integration tests

We've tested some of our logic, but can we also test the entire container that it works end-to-end. These tests are called "integration tests"

We already have some code for that test in `predict-test.py` which served us a sort-of integration test:

```python
import requests

url = 'http://localhost:9696/predict'

trip = {
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}

response = requests.post(url, json=trip).json()
print(response)
```

Now instead of just printing the response, let's add 
a few `assert`s:

```python
prediction = response['prediction']
assert 'duration' in prediction

assert 'version' in response
```

- We don't care what the prediction is as long as it's there 
- there should be the version field in the response

Let's make a copy and put it to the `integration_test` folder automate the test

Now we need to do this:

- Run docker build
- Run docker in detached mode (so the script can continue running)
- Run the rest
- Stop the docker container

```bash
docker build -t duration-prediction:integration-test ..
```

The `..` mean the context is one level up

```bash
PORT=9697
NAME="duration-prediction-integration-test"

docker run -it -d  \
    -p ${PORT}:9696 \
    --name="${NAME}" \
    duration-prediction:integration-test
```

- `-d` means we run in detached mode
- we assing a name with the `--name` parameter

We can see all running containers with `docker ps`

Let's now run the script. Since we use a different port, we need to modify it - to look in `URL` for the URL

```bash
export URL="http://localhost:${PORT}/predict"
python predict-test.py
```

Connecting to the container to see the logs:

```bash
docker logs ${NAME} --follow
```

Stopping and removing the container:

```bash
docker stop ${NAME} && docker rm ${NAME}
```

Now let's put everything together in `run.sh`:

```bash
#!/usr/bin/env bash

set -e

cd $(dirname $0)

PORT=9697

CONTAINER_NAME="duration-prediction-integration-test"

IMAGE_TAG="integration-test"
IMAGE_NAME="duration-prediction:${TAG}"

echo "building a docker image ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} ..

echo "building the image ${IMAGE_NAME}"
docker run -it -d  \
    -p ${PORT}:9696 \
    --name="${CONTAINER_NAME}" \
    ${IMAGE_NAME}

echo "sleeping for 3 seconds..."
sleep 3

echo "running the test..."
export URL="http://localhost:${PORT}/predict"
python predict-test.py

echo "test finished. logs:"
docker logs ${CONTAINER_NAME}

echo "stopping the container..."
docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}

echo "Done"
```

Note: a more complex (but more complete) script won't stop if
`predict-test` fails, it'll still show the logs. You can
google "saving error code to env variable in bash" and
adjust the script.


## Deployment to the Cloud

Outline

- AWS 
- User for the workshop
- AWS Elastic Beanstalk
- Alternatives


### AWS

Overview of the console and what AWS is. Create an account there if you haven't

### User for the workshop

For the workshop, we will create a user with permissions that
are only necessary for deployng to Elastic Beanstalk.

We should
avoid using keys with admin priviliges, because if the key
is compromised, you can have problems. Note that the user
we will create has VERY broad permissions too, so if it gets
compromised, we have problems too, so handle with care.


Creating a user and an access key

- IAM -> Access management -> Users
- Create user, e.g. "mle-workshop-day2", no need for "AWS Management Console"
- Attach the "AdministratorAccess-AWSElasticBeanstalk" policy
- Once created, open the user
- Go to security credentials, click "create access key"
- Use case: "Application running outside AWS" (doesn't matter much)
- Description: "codespaces" or "laptop" or something else
- Copy the key and the secret somewhere, we will use them
- You can deativate the key after the workshop if you want (access key -> actions -> deativate)

Now let's install AWS CLI:

```bash
pip install awscli
```

You can also install it as a dev dependency

```bash
pipenv install --dev awscli
```

Now configure the CLI and enter the key and the secret:

```bash
aws configure
```

Example:

```
AWS Access Key ID [None]: AKIA********
AWS Secret Access Key [None]: /N8Qh******
Default region name [None]: eu-west-1
Default output format [None]: 
```

Verify that it works:

```bash
aws sts get-caller-identity
```

Response:

```json
{
    "UserId": "AIDA********",
    "Account": "387546586099",
    "Arn": "arn:aws:iam::387546586099:user/mle-workshop-day2"
}
```

Now the user is ready

### AWS Elastic Beanstalk

First let's see EBS in console

After that, install the CLI:

```bash
pipenv install --dev awsebcli
```

Now it should be available as `eb`

Let's create an EBS project:

```bash
pipenv run eb init -p docker -r eu-west-1 duration-prediction
```

(replace the region with what you want)

To run locally, open `.elasticbeanstalk/config.yaml` and replace

```
  default_platform: 'Docker running on 64bit Amazon Linux 2'
  default_region: eu-west-1
```

Now run:

```bash
pipenv run eb local run --port 9696
```

And test it with `predict-test.py`:

```bash
export URL="http://localhost:9696/predict"
pipenv run python integration_tests/predict-test.py
```

It's time to deploy it:

```bash
pipenv run eb create duration-prediction-env
```

Note the URL:

```
2024-01-25 20:54:13    INFO    Application available at duration-prediction-env.eba-3t2zuiva.eu-west-1.elasticbeanstalk.com.
```

 Let's replace it:

```bash
export URL="http://duration-prediction-env.eba-3t2zuiva.eu-west-1.elasticbeanstalk.com/predict"
pipenv run python integration_tests/predict-test.py
```

It works!

Now let's terminate the EB environment

```bash
pipenv run eb terminate duration-prediction-env
```


### Alternatives

* Kubernetes (see mlzoomcamp.com if you want to learn more)
* ECS (AWS), Cloud Run (GCP)
* Many others


## Homework

- Serve the model you created yesterday


# Notes

- Usually we use separate repos for serving and training, each with its own CI/CD
- ...


# Links to the materials

This workshop is based on the courses and previous workshops I did:

- http://mlzoomcamp.com
- https://github.com/DataTalksClub/mlops-zoomcamp
- https://github.com/alexeygrigorev/lightweight-mlops-zoomcamp
- https://github.com/alexeygrigorev/hands-on-mlops-workshop

