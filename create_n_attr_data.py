import json
import itertools
import functools
import numpy as np
import random
from sklearn.model_selection import train_test_split

total_out = {}
total_out["attributes"] = ["Sex", "Glasses", "Haircolour", "Nose", "Mouth", "Eyecolour", "Facial Hair", "Hat", "Bald"]
total_out["props"] = {"Sex": ["Male", "Female"], "Glasses": ["Yes", "No"], "Haircolour": ["Brown", "Blond", "Black", "Red", "Gray"],
                "Nose": ["Big", "Small", "Regular"], "Mouth": ["Big", "Small", "Regular"], "Eyecolour": ["Blue", "Green", "Brown"],
                "Facial Hair": ["Beard", "Moustache", "Both", "None"], "Hat": ["Yes", "No"], "Bald":["Yes", "No"]}

for i in range(3, 10):
    out = {}
    out["attributes"] = total_out["attributes"][:i].copy()
    out["props"] = {}
    for attr in out["attributes"]:
        out["props"][attr] = total_out["props"][attr].copy()
    
    attrValVocab = functools.reduce(lambda x, y: x + y, out["props"].values())
    print(i, np.unique(np.array(attrValVocab)).shape)
    property_list = out["props"].values()
    combis = list(itertools.product(*property_list))
    combinations = []
    for combi in combis:
        combinations.append(list(combi))
    n_combis = len(combinations)
    train, test = train_test_split(combinations, test_size=0.1)
    train_combis = int(n_combis*0.9)
    test_combis = n_combis - train_combis
    out["numInst"] = {"train": train_combis, "test": test_combis}
    out["split"] = {"train": train, "test": test}
    out["taskDefn"] = [list(range(len(out["attributes"])))]
    print(i, out["taskDefn"])
    with open(f"data/who_is_it_{i}.json", 'w') as outfile:
        json.dump(out, outfile)

