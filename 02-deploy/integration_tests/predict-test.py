import os
import requests


default_url = 'http://localhost:9696/predict'
url = os.getenv('URL', default_url)


trip = {
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}

print(f'sending a POST request with {trip} to {url}')


response = requests.post(url, json=trip).json()
print(response)


prediction = response['prediction']
assert 'duration' in prediction
assert 'version' in response

print('All good')