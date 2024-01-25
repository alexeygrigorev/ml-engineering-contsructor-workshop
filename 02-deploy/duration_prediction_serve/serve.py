import os
import pickle

from flask import Flask, request, jsonify

from duration_prediction_serve.features import prepare_features


MODEL_PATH = os.getenv('MODEL_PATH', 'model.bin')
VERSION = os.getenv('VERSION', 'N/A')

with open(MODEL_PATH, 'rb') as f_in:
    model = pickle.load(f_in)


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'prediction': {
            'duration': pred,
        },
        'version': VERSION,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)