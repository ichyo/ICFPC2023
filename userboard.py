import requests
import os
import json

url = "https://api.icfpcontest.com/userboard"
token = os.environ['ICFPC_TOKEN']

r = requests.get(url, headers={'Authorization': 'Bearer ' + token})

results = r.json()['Success']['problems']

results = [x if x else 0.0 for x in results]

total = sum(results)

results_with_id = [(i + 1, x) for (i, x) in enumerate(results)]

cum = 0

for i, x in sorted(results_with_id, key=lambda x: x[1], reverse=True):
    prob = json.load(open(f'./problems/{i}.json'))
    m = len(prob['musicians'])
    a = len(prob['attendees'])
    p_num = (prob['stage_width'] - 19) * (prob['stage_height'] - 19)
    cum += x
    print(i, x, x / total, cum / total, m, a, p_num)



