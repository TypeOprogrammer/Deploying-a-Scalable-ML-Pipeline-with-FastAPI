import json
import requests

# Define the base URL for the API
url = "http://127.0.0.1:8000"

# Test GET request with basic error handling
try:
    r = requests.get(url)
    print("GET Status Code:", r.status_code)
    if r.status_code == 200:
        print("GET Response Message:", r.json())
    else:
        print(f"GET request failed with status code {r.status_code}")
except Exception as e:
    print(f"Error during GET request: {e}")

# Define the payload for the POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Test POST request
try:
    r = requests.post(f"{url}/data/", json=data)
    print(f"POST Status Code: {r.status_code}")
    if r.status_code == 200:
        print(f"Prediction Result: {r.json().get('result', 'No result key in response')}")
    else:
        print(f"POST request failed with status code {r.status_code}")
except Exception as e:
    print(f"Error during POST request: {e}")
