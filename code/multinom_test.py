import os
import json
import numpy as np
import math

vocab = np.load("./data/vocab.npy")
smooth_const = 1.0/len(vocab)

with open("./data/prob_y.json", 'r') as json_file:
    prob_y = json.load(json_file)

# change to p_v_y_bbow.json if bbow P(v|y) to be loaded
with open("./data/p_v_y_cbow.json", 'r') as json_file:
    p_v_y = json.load(json_file)

# change to bbow_test.json for bbow Multinomial classifier
with open("./data/cbow_test.json", 'r') as json_file:
    bow_dict = json.load(json_file)

parent_dir = "./20news-bydate/20news-bydate-test"
sub_dirs = os.listdir(parent_dir)
docs_y_dict = {}
for y in sub_dirs:
    docs_y_dict[y] = [s for s in bow_dict.keys() if s.split("/")[0]==y]

def get_prediction(test_file_path):
    token_dict = bow_dict[test_file_path]

    arg_max = -math.inf
    arg_ind = 0
    for y in sub_dirs:
        prod = prob_y[y]
        for token in token_dict:
            try:
                prod = prod + p_v_y[y][token] * token_dict[token]
            except:
                prod = prod + np.log(smooth_const)
        if arg_max < prod:
            arg_max = prod
            arg_ind = y

    return [arg_ind, arg_max]

# for prediction of a test doc, uncomment the below 2 lines and give value to test_file_path variable 
# test_file_path = "./20news-bydate/20news-bydate-test/comp.graphics/38758"
# print(get_prediction(test_file_path))
### gives [label, max value] --> dont uncomment this

def get_all_predictions(sub_dirs):
    pos = 0
    num_files = 0

    print("get all predictions")
    for y in sub_dirs:
        files_list = docs_y_dict[y]
        num_files += len(files_list)
        print(num_files)
         
        for each_file in files_list:
            [pred, value] = get_prediction(each_file)
            if pred == y:
                pos+=1
    return (1.0*pos)/num_files
     
# uncomment below line to get accuracy printed on terminal
# print(get_all_predictions(parent_dir, sub_dirs))
