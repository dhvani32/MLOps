import requests

data = {
    "feature1": 5.1,
    "feature2": 3.5,
    "feature3": 1.4,
    "feature4": 0.2
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print("Prediction:", response.json())