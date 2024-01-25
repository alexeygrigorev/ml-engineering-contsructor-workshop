MODEL_PATH = ./models/model-2022-01.bin
export MODEL_PATH

VERSION = 2022-01-v01
export VERSION


run:
	pipenv run python -m duration_prediction_serve.serve

.PHONY: tests
tests:
	pipenv run python -m unittest discover -s tests

docker_build: tests
	docker build -t duration-prediction .

docker_run: docker_build
	docker run -it -p 9696:9696 duration-prediction