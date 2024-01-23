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

- Notebooks for keeping experimental notebooks and one-off reports 
- Source code for the training pipeline
- Folder for tests
- Data (often use external data stores like s3/dvc - out of the scope)
- Models (often use model registries like mlflow - out of the scope)


So we will create the following folders:

```bash
mkdir src
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

Here we use argparse - a built-in python module. There are also packages like
[click](https://click.palletsprojects.com/en/8.1.x/) that make it even simpler. 

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

- Readme
- Documentation
- Logging
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


### Logging


### Testing


### Makefile


## Homework

- Refactor your own code


# Day 2: Deployment

Model --> Web service


## Serving with Flask

- Flask 101
- Building web services for ML models

## Dockerization

- Containerizing the web service


## Testing


## Deployment to the Cloud

- AWS Elastic Beanstalk.

## Homework

- Serve your model


# Notes

- Usually we use separate repos for serving and training, each with its own CI/CD
- ...