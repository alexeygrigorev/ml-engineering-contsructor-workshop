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

ENTRYPOINT [ \
    "gunicorn", \ 
    "--bind=0.0.0.0:9696", \ 
    "duration_prediction_serve.serve:app" \ 
]
