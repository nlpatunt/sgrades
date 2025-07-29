import requests

url = "http://localhost:8000/api/output-submissions"
files = {
    'file': open('sample_output.csv', 'rb')
}
data = {
    'dataset_id': 'asap-aes'
}
res = requests.post(url, files=files, data=data)
print(res.status_code)
print(res.json())
