import pandas as pd
import json


with open('data.json', 'r') as f:
  data = f.read()

info = json.loads(data)

df = pd.json_normalize(info["docs"])

df.to_csv("pool_info.csv")