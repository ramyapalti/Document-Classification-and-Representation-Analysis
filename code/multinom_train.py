import json
import numpy as np
import argparse
import os

# Initialize the Parser
parser = argparse.ArgumentParser(description ='Training Multinomial')

vocab = np.load("./data/vocab.npy")
parent_dir = "./20news-bydate/20news-bydate-train"
sub_dirs = os.listdir(parent_dir)

def save_p_v_y(bbow_flag, laplace_const):
    p_v_y = {}
    bow_dict = {}
    smooth_const = 1.0/len(vocab)
    with open("./data/" + bbow_flag + ".json", 'r') as json_file:
        bow_dict = json.load(json_file)

    docs_y_dict = {}
    for y in sub_dirs:
        docs_y_dict[y] = [s for s in bow_dict.keys() if s.split("/")[0]==y]

    p_v_y = {}
    print("numer calculation")
    num_i = 0
    denom = {}
    for y in sub_dirs:
        p_v_y[y]={}
    for y in sub_dirs:
        denom[y] = 0
        for v in vocab:
            num_i += 1
            if num_i % 1000 == 0:
                print(num_i)
            docs_y = docs_y_dict[y]
            numer = laplace_const 
            
            for each_doc in docs_y:
                try:
                    numer+=bow_dict[each_doc][v]

                except:
                    numer+=smooth_const
            denom[y] += numer
            p_v_y[y][v] = numer
            
    print("denom calculation")
    for y in p_v_y:
        for v in vocab:
            p_v_y[y][v] = np.log(p_v_y[y][v]/denom[y])

    with open("./data/p_v_y_" + bbow_flag + "_01.json", "w") as write_file:
        json.dump(p_v_y, write_file, indent=4)

def main():
    parser.add_argument('--type',  
                    dest = 'bbow_flag',
                    type = str,
                    help ='bbow flag')

    parser.add_argument('--laplace',
                        dest ='laplace_const', 
                        type = float,
                        help ='laplace constant')
    
    args = parser.parse_args()
    save_p_v_y(args.bbow_flag, args.laplace_const)

if __name__ == "__main__":
    main()                
