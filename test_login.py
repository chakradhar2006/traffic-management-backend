import requests
import time

url = "http://localhost:8000/api/login"
data = {"username": "Tejas20", "password": "@Tejas20@"}
print("Sending request...")
start_time = time.time()
try:
    response = requests.post(url, data=data, timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
print(f"Time taken: {time.time() - start_time:.2f} s")
