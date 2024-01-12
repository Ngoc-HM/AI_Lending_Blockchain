import requests

# url = "http://127.1.2.3:8000/rephrase"
# url = "http://127.2.3.1:8000/getTopK"
# url = "http://127.2.3.1:8000/getFinalAnswer"
url = "http://127.3.4.5:8002/chat"
question = input("Enter your question: ")

# payload = {"question": question}
payload = {"question": question}
# payload = {}
headers = {"Content-Type": "application/json", "accept": "application/json"}

response = requests.post(url, json = payload, headers=headers)
# with open("response.txt", "w") as file:
#     file.write(response.text)
print(response.json())