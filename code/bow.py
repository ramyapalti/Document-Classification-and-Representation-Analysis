import spacy
from spacy.lang.en import English
import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description ='Training Multinomial')

def get_all_files_dict(bbow_flag, parent_dir, dir_list, nlp):
    i = 0
    dict_all_files = {}
    for each_dir in dir_list:
        all_files = os.listdir(parent_dir + "/" +each_dir)
        for each_file in all_files:
            i+=1
            if i%100 == 0:
                print(i)
            doc_path = each_dir + "/" + each_file
            dict_file = get_dict_file(bbow_flag, parent_dir   + "/" + each_dir + "/", each_file, nlp)
            dict_all_files[doc_path] = dict_file
    return dict_all_files


def get_dict_file(bbow_flag, dir_name, file_name, nlp):
    fd = nlp(open(dir_name+file_name).read()) #load doc 
    tokens = [word.text.lower() for word in fd if word.text.isalnum()]
    dict_doc = {}
    if bbow_flag==1:
        for word in tokens:
            dict_doc[word]=1
    else:
       for word in tokens:
        try:
            dict_doc[word]+=1
        except:
            dict_doc[word]=1 
    return dict_doc


# dir_name = "./20news-bydate/20news-bydate-test"

def save_files(dir_path, bbow_file, cbow_file, nlp):
    dir_list = os.listdir(dir_path)
    dict_files_bbow = get_all_files_dict(1, dir_path, dir_list, nlp)
    with open(bbow_file, "w") as write_file:
        json.dump(dict_files_bbow, write_file, indent=4)
    dict_files_cbow = get_all_files_dict(0, dir_path, dir_list, nlp)
    with open(cbow_file, "w") as write_file:
        json.dump(dict_files_cbow, write_file, indent=4)

def main():
    parser.add_argument('--dir',  
                    dest = 'dir_path',
                    type = str,
                    help ='directory path')

    parser.add_argument('--bbow',
                        dest ='bbow_file', 
                        type = str)

    parser.add_argument('--cbow',
                        dest ='cbow_file', 
                        type = str)
    
    args = parser.parse_args()
    #load vocabulary
    nlp = spacy.load('en_core_web_sm')
    save_files(args.dir_path, args.bbow_file, args.cbow_file, nlp)

if __name__ == "__main__":
    main()                
