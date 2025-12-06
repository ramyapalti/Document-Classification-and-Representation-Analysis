import os
import json
import numpy as np

def get_vocab(bbow_file):
    bow_dict = {}
    with open(bbow_file, 'r') as json_file:
        bow_dict = json.load(json_file)

    vocab = []
    i = 0
    for each_file in bow_dict:
        i+=1
        if i%1000 == 0:
            print(i)
        vocab = vocab + list(bow_dict[each_file].keys())
    vocab = list(set(vocab))
    return vocab

#uncomment below 3 lines to save vocabulary file
# vocab = np.array(get_vocab("./data/cbow.json")) 
# with open('./data/vocab.npy', 'wb') as f:
#     np.save(f, vocab)

def save_p_y(parent_dir, p_y_file, sub_dirs):
    # save P(y) as json
    prob_y = {}
    total_len = 0
    for each_dir in sub_dirs:
        all_files = os.listdir(parent_dir + "/" +each_dir)
        total_len+=len(all_files)
        prob_y[each_dir] = len(all_files)

    for each_cnt in prob_y:
        prob_y[each_cnt] = np.log(prob_y[each_cnt]/total_len)

    with open(p_y_file, "w") as write_file:
        json.dump(prob_y, write_file, indent=4)

#uncomment below lines to save P(y) file 
# parent_dir = "./20news-bydate/20news-bydate-train"
# sub_dirs = os.listdir(parent_dir)
# p_y_file = "./data/prob_y.json"
# save_p_y(parent_dir, p_y_file, sub_dirs)