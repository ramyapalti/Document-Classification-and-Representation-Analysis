import os
import json
import numpy as np
import argparse

# Initialize the Parser
parser = argparse.ArgumentParser(description ='Training Poisson')

def save_files(alpha, beta):
    parent_dir = "./20news-bydate/20news-bydate-train"
    sub_dirs = os.listdir(parent_dir)

    # def save_lambda()
    vocab = np.load("./data/vocab.npy")
    # alpha = 0.5
    # beta = 0.25

    with open("./data/cbow.json", 'r') as json_file:
        bow_dict = json.load(json_file)

    docs_y_dict = {}
    for y in sub_dirs:
        docs_y_dict[y] = [s for s in bow_dict.keys() if s.split("/")[0]==y]

    lambda_v_y = {}
    for y in sub_dirs:
        lambda_v_y[y]={}

    words_seen = 0
    for v in vocab:
        words_seen += 1
        if words_seen%1000 == 0:
            print(words_seen)
        for y in sub_dirs:
            cnt_v = 0
            for each_doc in docs_y_dict[y]:
                try:
                    cnt_v += bow_dict[each_doc][v]
                except:
                    continue
            lambda_v_y[y][v] = (cnt_v + alpha)/(len(docs_y_dict[y]) + beta)

    with open("./data/lambda_v_y.json", "w") as write_file:
        json.dump(lambda_v_y, write_file, indent=4)


def main():
    parser.add_argument('--alpha',  
                    dest = 'alpha',
                    type = float)

    parser.add_argument('--beta',
                        dest ='beta', 
                        type = float)

    args = parser.parse_args()
    
    save_files(args.alpha, args.beta)

if __name__ == "__main__":
    main() 