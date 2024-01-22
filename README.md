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

## Environment Setup

Overview:

- GitHub Codespaces
- Installing the required packages
- Alternative setups

Note: You don't have to use GitHub codespaces, you can do it locally

### Creating a codespace

![image](https://github.com/alexeygrigorev/ml-engineering-contructor-workshop/assets/875246/15529150-74a5-4295-9f5c-a9de857b6ac6)

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

![image](https://github.com/alexeygrigorev/ml-engineering-contructor-workshop/assets/875246/760c27a7-6dc7-4486-a0fd-c0169e81ca60)

- Codespaces -> Stop current codespace
- If you need to keep it: codespaces -> select the code space -> untick "auto-delete codespace"


Links:

- [Codespaces overview](https://docs.github.com/en/codespaces/overview)
- [Billing](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces)
- Further learning: [Python Dev Containers](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/setting-up-your-python-project-for-codespaces)



## Notebook to Script Transformation

- Converting and parameterizing a Jupyter notebook into a Python script
- Job parametrization
- Saving the model for deployment


### Problem

Taxi trip duration prediction

- Dataset: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Goal: predict the time to travel from point A to point B


### Preparation

* We'll start with the model [we already created previously](01-train/duration-prediction-starter.ipynb)
* Copy this notebook to "duration-prediction.ipynb"
* This model is used for preducting the duration of a taxi trip

### Preparing the environment

Create the enviroment

```bash
pipenv install
```

Install the packages:

```bash
pipenv install scikit-learn==1.2.2 pandas pyarrow
pipenv install --dev jupyter
```

Dev dependencies: only for local environment, not deployed to production

On Linux you might also need to instal `pexpect` for jupyter:

```bash
pipenv install --dev jupyter pexpect
```

We will use the data from the [NYC TLC website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

* Train: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
* Validation: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet

Run the notebook

```bash
pipenv run jupyter notebook
```

Forward the 8888 port if you're running the code remotely

- In VS Code: Ports -> Forward a Port

Now open http://localhost:8888/


### Project Structure

- Best practices in structuring ML training pipelines
- Cookie-cutter

## Best practices

- Documentation
- Logging
- Testing

## Homework

- Refactor your own code


# Day 2: Deployment

Model --> Web service

## Serving with Flask

- Flask 101
- Building web services for ML models

## Dockerization

- Containerizing the web service

## Deployment to the Cloud

- AWS Elastic Beanstalk.

## Homework

- Serve your model

