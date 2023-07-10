import requests
import os
import json

token = os.environ['ICFPC_TOKEN']


#18

for id in range(19, 90 + 1):
    print(id)
    url = "https://api.icfpcontest.com/submissions?offset=0&limit=1000&problem_id=" + str(id)
    r = requests.get(url, headers={'Authorization': 'Bearer ' + token})
    r.raise_for_status()
    data = r.json()
    v = data["Success"]
    sv = [x for x in v if 'Success' in x['score']]
    if len(sv) == 0:
        print("Skip " + str(id))
        continue
    print(sv)
    # sub = max(sv, key=lambda x:x['score']['Success'])

    # url = "https://api.icfpcontest.com/submission?submission_id=" + sub['_id']
    # r = requests.get(url, headers={'Authorization': 'Bearer ' + token})
    # r.raise_for_status()
    # data = r.json()
    # contents_str = data['Success']['contents']
    # contents = json.loads(contents_str)
    # m_num = len(contents['placements'])

    # assert 'volumes' in contents
    # assert contents['volumes'] == [10.0] * m_num

    # contents['volumes'] = [10.0] * m_num

    # submission = {
    #     'problem_id': id,
    #     'contents': json.dumps(contents),
    # }

    # url = "https://api.icfpcontest.com/submission"
    # r = requests.post(url, json=submission, headers={'Authorization': 'Bearer ' + token})
    # r.raise_for_status()



