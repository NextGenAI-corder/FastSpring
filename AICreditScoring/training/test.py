import pickle
with open("models/cat_maps.pkl", "rb") as f: cat_maps = pickle.load(f)
print(cat_maps)