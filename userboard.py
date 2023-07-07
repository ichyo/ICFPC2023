import requests
import os

url = "https://api.icfpcontest.com/userboard"
token = os.environ['ICFPC_TOKEN']

r = requests.get(url, headers={'Authorization': 'Bearer ' + token})

results = r.json()['Success']['problems']

results = [x if x else 0.0 for x in results]

total = sum(results)

results_with_id = [(i + 1, x) for (i, x) in enumerate(results)]

for i, x in sorted(results_with_id, key=lambda x: x[1], reverse=True):
    print(i, x, x / total)



