# Duration Prediction Web Service

## Brief Description
This web service uses a machine learning model to predict the duration of a trip based on various inputs. It is built using Flask and deployed using Docker and AWS.

## Installation Instructions
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mle-workshop-day2-deploy.git
   ```
2. Navigate to the project directory:
   ```
   cd mle-workshop-day2-deploy
   ```
3. Install dependencies:
   ```
   pip install pipenv
   pipenv install flask scikit-learn==1.2.2
   ```

## Dependencies
- Python 3.10
- Flask
- Scikit-Learn 1.2.2
- Docker for containerization
- AWS for deployment

## Configuration
Set environment variables for the model path and version:
```
export MODEL_PATH="./models/model-2022-01.bin"
export VERSION="2022-01-v01"
```

## Structure
- `models/`: Contains the machine learning model files
- `duration_prediction_serve/`: Flask app files
- `tests/`: Unit tests
- `integration_tests/`: Integration tests
- `Dockerfile`: Defines the Docker container configuration
- `Makefile`: Automation commands
- `README.md`: Project documentation

## Usage
### Running the Flask App
To run the Flask application locally:
```
pipenv run python duration_prediction_serve/serve.py
```
This will start the web service on `http://localhost:9696`.

### Testing the Flask App with Requests
After starting the Flask app, you can test it using `curl` or any HTTP client. Example using `curl`:
```
curl -X POST -H "Content-Type: application/json" -d '{"PULocationID": 100, "DOLocationID": 102, "trip_distance": 30}' http://localhost:9696/predict
```
This should return a JSON response with the prediction.

### Running Tests
To run unit tests, navigate to the project directory and execute:
```
pipenv run pytest tests/
```

### Running Integration Tests
Ensure the Flask app is running as a separate process or in a Docker container, then execute:
```
pipenv run python integration_tests/predict-test.py
```
This script will send requests to the Flask app and assert the responses.

## Deployment
The project is containerized using Docker and deployed on AWS. The `Dockerfile` contains all necessary instructions for building the image.

## Access
If deployed, the web service can be accessed via the URL provided by the AWS deployment.

## Documentation
For more detailed information, refer to external documentation linked here.

## Contributing
Guidelines for contributing to this project are as follows:
- Fork the repository
- Create a feature branch
- Commit your changes
- Push to the branch
- Open a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Credits to third-party resources or contributors that were used in this project.
