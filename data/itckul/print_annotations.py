import pickle

with open('itckul_infos_train.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)  # Inspect the structure and check if labels and masks are correctly stored
