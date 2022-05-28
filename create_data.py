import json
import itertools
import random
from sklearn.model_selection import train_test_split

out = {}
n_combis=800
out["attributes"] = ["Sex", "Glasses", "Haircolour", "Nose", "Mouth", "Eyecolour", "Facial Hair", "Hat", "Bald"]
out["props"] = {"Sex": ["Male", "Female"], "Glasses": ["Yes", "No"], "Haircolour": ["Brown", "Blond", "Black", "Red", "Gray"],
                "Nose": ["Big", "Small", "Regular"], "Mouth": ["Big", "Small", "Regular"], "Eyecolour": ["Blue", "Green", "Brown"],
                "Facial Hair": ["Beard", "Moustache", "Both", "None"], "Hat": ["Yes", "No"], "Bald":["Yes", "No"]}
property_list = out["props"].values()
combis = list(itertools.product(*property_list))
combinations = []
for combi in combis:
    if combi[0] == "Female" and combi[6] != "None":
        continue
    else:
        combinations.append(list(combi))
combinations = random.sample(combinations, n_combis)
train, test = train_test_split(combinations, test_size=0.1)
train_inst = int(n_combis*0.9)
test_inst = n_combis - train_inst
out["numInst"] = {"train": train_inst, "test": test_inst}
out["split"] = {"train": train, "test": test}
out["taskDefn"] = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
with open("data/who_is_it.json", 'w') as outfile:
    json.dump(out, outfile)

