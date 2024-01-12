from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
coin_name="ethereum"
url = 'https://develop.centic.io/dev/v3/ranking/defi?category=Lending'
headers = {
            'Accepts': 'application/json',
            'X-Apikey': 'QN1qnzREpjrLQ45aw5VMHGPh1UVo9Ka72XGEExYOmC3TEOCt',
            }
parameters={}
session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)

  with open("/home/xantus/Landing_chatbot/data.json", 'w') as json_file:
    json.dump(data, json_file)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)