#!/bin/bash

for id in {1..45}
do
    wget "https://cdn.icfpcontest.com/problems/$id.json" -P ./problems -nc 
    # curl "https://api.icfpcontest.com/problem?problem_id=$id" | jq -r .Success > ./problems/$id.json
done


