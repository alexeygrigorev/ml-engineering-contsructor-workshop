#!/usr/bin/env bash

set -e

cd $(dirname $0)

PORT=9697

CONTAINER_NAME="duration-prediction-integration-test"

IMAGE_TAG="integration-test"
IMAGE_NAME="duration-prediction:${IMAGE_TAG}"

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
