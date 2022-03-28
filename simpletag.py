#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 16:42:06 2021

@author: lizhenxing
"""
#%% import pkg
import random
# import math
import operator
import pandas as pd
import time

#%% load in data and create var
file_path = "./user_taggedbookmarks-timestamps.dat"
records = {}
train_data = dict()
test_data = dict()
user_tags = dict()
tag_items = dict()
user_items = dict()

#%% use test_dataset to derive accuracy and recall
def precisionAndRecall(N):
    hit = 0
    h_recall = 0
    h_precision = 0
    for user, items in test_data.items():
        if user not in train_data:
            continue
        rank = recommend(user, N)
        for item, rui in rank:
            if item in items:
                hit += 1
        h_recall += len(items)
        h_precision += N
    h_recall = hit/(h_recall*1.0)
    h_precision = hit/(h_precision*1.0)
    return (h_precision, h_recall)

#%% recommend topN to the users
def recommend(user, N):
    recommend_items = dict()
    tagged_items = user_items[user]
    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue
            if item not in recommend_items:
                recommend_items[item] = wut * wti
            else:
                recommend_items[item] = recommend_items[item] + wut * wti
    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0:N]

#%% use the test_dataset to evaluate the recommendation results
def testRecommend():
    print("推荐结果评估")
    print("%3s %10s %10s" % ('N', "精确率","召回率"))
    for n in [5,10,20,40,60,80,100]:
        precision, recall = precisionAndRecall(n)
        print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))
        
#%% load in data
def load_data():
    print("开始加载数据...")
    df = pd.read_csv(file_path, sep='\t')
    for i in range(len(df)):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tag = df['tagID'][i]
        records.setdefault(uid,{})
        records[uid].setdefault(iid,[])
        records[uid][iid].append(tag)
    print("数据集大小为 %d." % (len(df)))
    print("设置tag的人数 %d." % (len(records)))
    print("数据加载完成\n")
        
#%% dividing the dataset into train_set and test_set
def train_test_split(ratio, seed=100):
    random.seed(seed)
    for u in records.keys():
        for i in records[u].keys():
            if random.random()<ratio:
                test_data.setdefault(u,{})
                test_data[u].setdefault(i,[])
                for t in records[u][i]:
                    test_data[u][i].append(t)
            else:
                train_data.setdefault(u,{})
                train_data[u].setdefault(i,[])
                for t in records[u][i]:
                    train_data[u][i].append(t)
    print("训练集样本数 %d, 测试集样本数 %d" % (len(train_data),len(test_data)))

    
#%% set matrix mat[index, item] = 1
def addValueToMat(mat, index, item, value=1):
    if index not in mat:
        mat.setdefault(index,{})
        mat[index].setdefault(item,value)
    else:
        if item not in mat[index]:
            mat[index][item] = value
        else:
            mat[index][item] += value

#%% use train_set to initialize user_tags, tag_items, user_items
def initSata():
    records = train_data
    for u, items in records.items():
        for i, tags in items.items():
            for tag in tags:
                addValueToMat(user_tags, u, tag, 1)
                addValueToMat(tag_items, tag, i, 1)
                addValueToMat(user_items, u, i, 1)
    print("user_tags, tag_items, user_items初始化完成。")
    print("user_tags大小 %d, tag_items大小 %d, user_items大小 %d" % (len(user_tags), len(tag_items), len(user_items)))
    
#%% main 
start = time.time()
load_data()
mid1 = time.time()
train_test_split(0.2)
mid2 = time.time()    
initSata()    
mid3 = time.time()
testRecommend()        
end = time.time()

print('load_data time: ', '%.5f'%(mid1 - start))            # split: 0.2, about 5s 
print('split time: ', '%.5f'%(mid2 - mid1))                 # split: 0.2, about 0.16s
print('training time: ', '%.5f'%(mid3 - mid2))              # split: 0.2, about 0.5s
print('testing time: ', '%.5f'%(end - mid3))                # split: 0.2, about 130s
# %%
