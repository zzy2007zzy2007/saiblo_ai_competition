import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else r"D:\dsh_build_tmp\lowq_smoke2\az_selfplay_seed00000.ents.jsonl"
rows = []
with open(p, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
print("total rows:", len(rows))
for idx in (100, 300, 600):
    if idx < len(rows):
        r = rows[idx]
        print("--- row", idx,
              "round=", r["round"], "player=", r["player"],
              "hp=", r["hp_us"], r["hp_opp"], "coins=", r["coins"],
              "die=", r["die"], "old=", r["old"],
              "nT=", len(r["towers"]), "nA=", len(r["ants"]),
              "eff=", len(r["effects"]), "val=", r["value_target"])
        if r["towers"]:
            print("    tower0:", r["towers"][0])
        if r["ants"]:
            print("    ant0:", r["ants"][0])
# 覆盖度
if rows:
    print("round range:", rows[0]["round"], "-", max(x["round"] for x in rows))
    print("max nT:", max(len(x["towers"]) for x in rows), "max nA:", max(len(x["ants"]) for x in rows))
    print("die nonzero rows:", sum(1 for x in rows if x["die"] != [0, 0]),
          "old nonzero rows:", sum(1 for x in rows if x["old"] != [0, 0]))
