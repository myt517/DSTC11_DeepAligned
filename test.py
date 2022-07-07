import argparse
import csv
import io
import json
import os
import numpy as np

import requests
from tqdm import tqdm

def main():
    text = []
    intent_label = []
    frame = {}
    ratio = ""
    path = "./data/HWU64/raw.json"
    with open(path, "r") as json_file:
        json_dict = json.load(json_file)

    all = json_dict["utterances"]
    for i in range(len(all)):
        text.append(all[i]["text"])
        intent_label.append(all[i]["intent"])

    label_list = np.unique(np.array(intent_label))
    print(len(label_list))

    for k in label_list:
        if k in frame.keys():
            for i in range(len(text)):
                if intent_label[i] == k:
                    frame[k].append(text[i])
        else:
            frame[k] = []
            for i in range(len(text)):
                if intent_label[i] == k:
                    frame[k].append(text[i])

    max_len = 0
    for i in range(len(text)):
        a = len(text[i].split(" "))
        print(text[i])
        print(a)
        exit()
        if max_len < a:
            max_len = a

    print(max_len)
    exit()



    total_num = 0
    for k in label_list:
        num = len(frame[k])
        total_num+=num

    file_name_train = "./data/HWU64/train.tsv"
    file_name_dev = "./data/HWU64/dev.tsv"
    file_name_test = "./data/HWU64/test.tsv"

    f1 = open(file_name_train, 'w', encoding='utf-8')
    f2 = open(file_name_dev, 'w', encoding='utf-8')
    f3 = open(file_name_test, 'w', encoding='utf-8')

    csv_writer = csv.writer(f1, delimiter='\t')
    csv_writer.writerow(["text", "label"])

    for k in frame.keys():
        num = len(frame[k])
        for i in range(0,int(num*0.6)):
            csv_writer.writerow([frame[k][i], k])

    f1.close()

    print("---------------------")

    csv_writer = csv.writer(f2, delimiter='\t')
    csv_writer.writerow(["text", "label"])

    for k in frame.keys():
        num = len(frame[k])
        for i in range(int(num * 0.6), int(num * 0.8)):
            csv_writer.writerow([frame[k][i], k])

    f2.close()

    print("---------------------")

    csv_writer = csv.writer(f3, delimiter='\t')
    csv_writer.writerow(["text", "label"])

    for k in frame.keys():
        num = len(frame[k])
        for i in range(int(num * 0.8), num):
            csv_writer.writerow([frame[k][i], k])

    f1.close()




if __name__ == "__main__":
    main()