def process_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        txt = file.read()

    fse = txt.split()[36:]  # Skip header
    labs = dict()
    for i in range(0, len(fse), 3):
        si = int(fse[i])
        ei = int(fse[i+1])
        flab = fse[i+2][0]
        slab = fse[i+2][1]
        
        if flab == 'N' and slab == 'N':
            lab = 'No Fire No Smoke'
        elif flab == 'Y' and slab == 'N':
            lab = 'Fire No Smoke'
        elif flab == 'Y' and slab == 'Y':
            lab = 'Fire Smoke'
            
        for j in range(si, ei+1):
            labs[j] = lab
            
    return labs

def get_stratified_sample(labs, n_samples=1000, random_state=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    
    keys = np.array(list(labs.keys()))
    labels = np.array(list(labs.values()))
    
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
    for _, sample_index in stratified_split.split(keys, labels):
        stratified_sample_keys = keys[sample_index]
        stratified_sample_labels = labels[sample_index]
        
    stratified_sample_dict = {key: label for key, label in zip(stratified_sample_keys, stratified_sample_labels)}
    return stratified_sample_dict 