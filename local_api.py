import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000/") # Your code here
print("Status Code:", r.status_code)
print("Result:", r.json().get("message"))
# TODO: print the status code
# print()
# TODO: print the welcome message
# print()



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


# TODO: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/",json=data)
                 
print("Status Code:", r.status_code)
try:
    print("Result:", r.json().get("result"))
except Exception:
    print("Body:",r.text)
     
# Your code here

# TODO: print the status code
# print()
# TODO: print the result
# print()
