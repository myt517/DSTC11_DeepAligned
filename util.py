import itertools
import subprocess
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from typing import List
import torch
import copy
import torch.nn.functional as F
import random
import csv,json
import sys
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn import metrics
matplotlib.use('Agg')


CLINC_domian_intent = {
    "banking": ["freeze_account",
"routing",
"pin_change",
"bill_due",
"pay_bill",
"account_blocked",
"interest_rate",
"min_payment",
"bill_balance",
"transfer",
"order_checks",
"balance",
"spending_history",
"transactions",
"report_fraud"],
    "credit_cards": ["replacement_card_duration",
"expiration_date",
"damaged_card",
"improve_credit_score",
"report_lost_card",
"card_declined",
"credit_limit_change",
"apr",
"redeem_rewards",
"credit_limit",
"rewards_balance",
"application_status",
"credit_score",
"new_card",
"international_fees"],
    "kitchen_and_dining": ["food_last",
"confirm_reservation",
"how_busy",
"ingredients_list",
"calories",
"nutrition_info",
"recipe",
"restaurant_reviews",
"restaurant_reservation",
"meal_suggestion",
"restaurant_suggestion",
"cancel_reservation",
"ingredient_substitution",
"cook_time",
"accept_reservations"],
    "home": ["what_song",
"play_music",
"todo_list_update",
"reminder",
"reminder_update",
"calendar_update",
"order_status",
"update_playlist",
"shopping_list",
"calendar",
"next_song",
"order",
"todo_list",
"shopping_list_update",
"smart_home"],
    "work": ["pto_request_status",
"next_holiday",
"insurance_change",
"insurance",
"meeting_schedule",
"payday",
"taxes",
"income",
"rollover_401k",
"pto_balance",
"pto_request",
"w2",
"schedule_meeting",
"direct_deposit",
"pto_used"],
    "utility": ["weather",
"alarm",
"date",
"find_phone",
"share_location",
"timer",
"make_call",
"calculator",
"definition",
"measurement_conversion",
"flip_coin",
"spelling",
"time",
"roll_dice",
"text"],
    "travel": ["plug_type",
"travel_notification",
"translate",
"flight_status",
"international_visa",
"timezone",
"exchange_rate",
"travel_suggestion",
"travel_alert",
"vaccines",
"lost_luggage",
"book_flight",
"book_hotel",
"carry_on",
"car_rental"],
    "auto_and_commute": ["current_location",
"oil_change_when",
"oil_change_how",
"uber",
"traffic",
"tire_pressure",
"schedule_maintenance",
"gas",
"mpg",
"distance",
"directions",
"last_maintenance",
"gas_type",
"tire_change",
"jump_start"],
    "small_talk": ["who_made_you",
"meaning_of_life",
"who_do_you_work_for",
"do_you_have_pets",
"what_are_your_hobbies",
"fun_fact",
"what_is_your_name",
"where_are_you_from",
"goodbye",
"thank_you",
"greeting",
"tell_joke",
"are_you_a_bot",
"how_old_are_you",
"what_can_i_ask_you"],
    "meta": ["change_speed",
"user_name",
"whisper_mode",
"yes",
"change_volume",
"no",
"change_language",
"repeat",
"change_accent",
"cancel",
"sync_device",
"change_user_name",
"change_ai_name",
"reset_settings",
"maybe"],



}


def get_imbalanced(OOD_list):
    OOD_list_ranking = []
    for i in range(12):
        tmp_list = random.sample(OOD_list, 5)
        OOD_list = list(set(OOD_list).difference(set(tmp_list)))
        OOD_list_ranking.append(tmp_list)

    print(OOD_list_ranking)

    #for i in range(12):
    #    for k in range(len(OOD_list_ranking[i])):
    #        print(OOD_list_ranking[i][k])

    return OOD_list_ranking


def imbalanced_division(train_OOD, OOD_list_ranking):
    train_OOD_selected = []

    for i in range(len(OOD_list_ranking)):
        for k in range(len(OOD_list_ranking[i])):
            label = OOD_list_ranking[i][k]
            samples = [example for example in train_OOD if example.label == label]
            num_samples = random.randint((i + 1) * 10 - 9, (i + 1) * 10)
            print(num_samples)
            samples_selected = random.sample(samples, num_samples)
            train_OOD_selected.extend(samples_selected)


    return train_OOD_selected

def IND_division(train_IND, IND_list):

    selected_examples = []
    for i in range(len(IND_list)):
        label = IND_list[i]
        samples = [example for example in train_IND if example.label == label]
        selected_examples.extend(samples)
    print(len(selected_examples))

    file_name = './data/clinc/IND_0.8.tsv'
    write_csv(selected_examples, file_name)

def labeled_division(train_IND, file_name):
    write_csv(train_IND, file_name)

def write_csv(train_IND_selected, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(f, delimiter='\t')
    csv_writer.writerow(["text", "label"])

    for i in range(len(train_IND_selected)):
        csv_writer.writerow([train_IND_selected[i].text_a, train_IND_selected[i].label])

    f.close()



def get_oos(ratio=0.1):
    oos_file_path="./data/clinc/data_oos_plus.json"
    oos_list = []
    with open(oos_file_path, 'r') as f:
        data_frame = json.load(f)
    print(len(data_frame["oos_train"]))
    print(len(data_frame["oos_test"]))
    print(len(data_frame["oos_val"]))
    print(data_frame["oos_train"][0][0])
    print(data_frame["oos_train"][0][1])

    for i in range(len(data_frame["oos_train"])):
        oos_list.append(data_frame["oos_train"][i])

    for i in range(len(data_frame["oos_test"])):
        oos_list.append(data_frame["oos_test"][i])

    for i in range(len(data_frame["oos_val"])):
        oos_list.append(data_frame["oos_val"][i])
    print(len(oos_list))
    #num_samples = round(len((oos_list)) * ratio)
    num_samples = 360
    oos_list_selected = random.sample(oos_list, num_samples)

    #print(oos_list_selected)


    return oos_list_selected


def cross_domain_division(IND_domains, OOD_domain = None):

    IND_class, OOD_class = [], []
    for d in IND_domains:
        IND_class.extend(CLINC_domian_intent[d])

    if OOD_domain != None:
        for d in OOD_domain:
            OOD_class.extend(CLINC_domian_intent[d])

        return IND_class, OOD_class

    return  IND_class

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def cluster_F1(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    F1_score = f1_score(y_true, y_pred_aligned, average='weighted')
    return F1_score

def cluster_precision(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    precision = precision_score(y_true, y_pred_aligned, average='weighted')
    return precision

def cluster_recall(y_true, y_pred):
    ind, _ = hungray_aligment(y_true, y_pred)
    map_ = {i[0]: i[1] for i in ind}
    y_pred_aligned = np.array([map_[idx] for idx in y_pred])
    recall = recall_score(y_true, y_pred_aligned, average='weighted')
    return recall

def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2),
            'F1': round(f1_score(y_true, y_pred, average='weighted')*100, 2),
            'PRE': round(precision_score(y_true, y_pred, average='weighted')*100, 2),
            'REC': round(recall_score(y_true, y_pred, average='weighted')*100, 2),
            }


def intra_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    #print(len(cluster_center))

    intra_cluster_distance = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        dist = np.sqrt(np.sum(np.square(X_feats - cluster_center[i]), axis=1))
        dist = dist.tolist()
        intra_cluster_distance.append(dist)

    min_list, max_list, mean_list = [], [], []
    for i in range(len(intra_cluster_distance)):
        min_list.append(min(intra_cluster_distance[i]))
        max_list.append(max(intra_cluster_distance[i]))
        mean_list.append(sum(intra_cluster_distance[i])/len(intra_cluster_distance[i]))


    min_d = min(min_list)
    max_d = max(max_list)
    mean_d = sum(mean_list)/len(mean_list)

    return min_d, max_d, mean_d


def inter_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        #print(X_feats)
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    print(len(cluster_center), cluster_center[0].shape)

    #print(cluster_center)
    #print("______________________________________")

    #knn.fit(cluster_center, labels)
    #y_pred = knn.predict(cluster_center)
    #print(y_pred)

    inter_cluster_distance = []
    for i in range(len(cluster_center)):
        dist_list = []
        for j in range(len(cluster_center)):
            dist = np.sqrt(np.sum(np.square(cluster_center[i] - cluster_center[j])))
            dist_list.append(dist)
        inter_cluster_distance.append(dist_list)

    inter_distance = []
    for i in range(len(inter_cluster_distance)):
        inter_cluster_distance[i].sort()
        #print(inter_cluster_distance[i])
        tmp = np.mean(inter_cluster_distance[i][1:3])
        inter_distance.append(tmp)

    inter_distance = np.array(inter_distance)
    print(inter_distance.shape)

    # Min
    min_d = np.float(inter_distance.min())
    # Max
    max_d = np.float(inter_distance.max())
    # Mean
    mean_d = np.float(inter_distance.mean())


    return min_d, max_d, mean_d


def TSNE_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    color_list = ["blueviolet", "green", "blue", "yellow", "purple", "black", "brown", "cyan", "gray", "pink", "orange",
                  "red", "greenyellow", "sandybrown", "deeppink", 'olive', 'm', 'navy']

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    i = 0
    for _class in classes:
        if _class == "unseen":
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=i, alpha=0.5, s=6, edgecolors='none', zorder=15, color = color_list[i])
        i+=1
    ax.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    fig.subplots_adjust(right=0.8)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")
    #plt.savefig(save_path, format="png")

    print()
    

