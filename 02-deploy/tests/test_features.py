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
