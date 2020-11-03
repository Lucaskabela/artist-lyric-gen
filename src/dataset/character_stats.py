import json

with open("character_stats.json") as openfile:
    stats = json.load(openfile)
    print(stats)