# Taxi Trip Duration Prediction - Training Pipeline

This project focuses on converting a Jupyter notebook into a training pipeline to predict taxi trip durations. It utilizes the NYC TLC Trip Record Data to model and forecast travel times in New York City.

## Installation instructions

Pre-requisites:

- Docker
- Python 3.10
- Pipenv


```bash
# note that the address is fake
git clone https://github.com/alexeygrigorev/mle-workshop-day1-train.git
pipenv install --dev
```

## Project Structure

The project is organized into several directories and files, each serving a specific purpose:

- `src/`: the source code for the training pipeline and other Python scripts
- `tests/`: test cases and test data
- `notebooks/`: notebooks for experimentation and one-off analyses
- `data/`: data files used for training and validating the models
- `models/`: Where trained model files are saved
- `README.md`: The main documentation file providing an overview and instructions for the project
- `Makefile`: A script for automating common tasks like testing, running, or deploying the project


## Usage

You need to be in the `src` folder

```bash
cd src
```

Running it:

```python

TRAIN="2022-01"
VAL="2022-02"

pipenv run python train.py \
    --train-month="${TRAIN}" \
    --validation-month="${VAL}" \
    --model-output-path="../models/model-${TRAIN}.bin"
```

Running tests:

```bash
TODO
```
 