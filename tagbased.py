# -*- coding: utf-8 -*-
"""
Created on 2020/11/18 13:54

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import operator
import time
import random
import pandas as pd
import numpy as np

class TagBased(object):
    def __init__(self, data_path, sep='\t'):
        self.data_path = data_path
        self.sep = sep
        self.calc_result = {}
        self.table = self.__load_data(data_path, sep)

    def __load_data(self, data_path, sep):
        table = pd.read_table(data_path, sep=sep)
        return table

    def __calc_frequency(self, table):
        # user -> item
        user_item = table.groupby(by=['userID','bookmarkID'])['tagID'].count()
        # user -> tag
        user_tag = table.groupby(by=['userID', 'tagID'])['bookmarkID'].count()
        # tag -> item
        tag_item = table.groupby(by=['tagID', 'bookmarkID'])['userID'].count()
        # tag -> user
        tag_user = table.groupby(by=['tagID', 'userID'])['bookmarkID'].count()
        return {"user_item": user_item, "user_tag": user_tag, "tag_item": tag_item, "tag_user": tag_user}

    def train_test_split(self, ratio, seed):
        return self.__train_test_split(self.table, ratio, seed)

    def __train_test_split(self, table, ratio, seed):
        random.seed(seed)
        t1 = time.time()
        stratify_count = table.groupby(by='userID')['userID'].count()
        stratify_df = pd.DataFrame({"count":stratify_count})
        stratify_df['test_num'] = (stratify_df['count'] * ratio ).apply(int)
        test_id = []
        train_id = []
        """
        ==========================================
        # 方法 1：dataframe的iterrows遍历行
        # 10次执行平均耗时约 2.3秒
        ==========================================
        # for index, row in stratify_df.iterrows():
        #     tmp_ids = table[table['userID'] == index].index.tolist()
        #     tmp_test_id = random.sample(tmp_ids, row['test_num'])
        #     test_id.extend(tmp_test_id)
        #     train_id.extend(list(set(tmp_ids) -set(tmp_test_id)))

        ==========================================
        # 方法 2：series map和dataframe apply
        # 10次执行平均耗时约 2.2秒
        ==========================================
        按理来说，apply + map会比 iterrows 快很多，可能下面这个写法存储了较多的list，主要是为了方便查看拆分的结果，
        所以在遍历取数的时候会花费更多的时间。
        虽然apply+map方法只比iterrows快了0.1秒左右，但是在写法上我还是喜欢用apply+map。
        """
        stratify_df['ids'] = stratify_df.index.map(lambda x: table[table['userID'] == x].index.tolist())
        stratify_df['test_index'] = stratify_df.apply(lambda x: random.sample(x['ids'], x['test_num']), axis=1)
        stratify_df['train_index'] = stratify_df.apply(lambda x: list(set(x['ids']) - set(x['test_index'])), axis=1)
        stratify_df['test_index'].apply(lambda x: test_id.extend(x))
        stratify_df['train_index'].apply(lambda x: train_id.extend(x))
        train_data = table.iloc[train_id].reset_index(drop=True)
        test_data = table.iloc[test_id].reset_index(drop=True)
        print("Split train test dataset by stratification, time took: %.4f" % (time.time() - t1))
        return {"train_data": train_data, "test_data": test_data}

    def __calc_item_recommendation(self, user_id, user_item, user_tag, tag_item, n, method):
        marked_item = user_item[user_id].index
        recommend = {}
        # t1 = time.time()
        # user_id -> tag -> item -> count
        marked_tag = user_tag.loc[user_id]
        marked_tag_sum = marked_tag.values.sum()
        for tag_index, tag_count in marked_tag.iteritems():
            selected_item = tag_item.loc[tag_index]
            selected_item_sum = selected_item.values.sum()
            tag_selected_users_sum = self.calc_result['tag_user'].loc[tag_index].values.sum()
            for item_index, tag_item_count in selected_item.iteritems():
                if item_index in marked_item:
                    continue
                if item_index not in recommend:
                    if method == 'norm':
                        recommend[item_index] = (tag_count / marked_tag_sum) * (tag_item_count / selected_item_sum)
                    elif method == 'simple':
                        recommend[item_index] = tag_count * tag_item_count
                    elif method == 'tfidf':
                        recommend[item_index] = tag_count / np.log(1 + tag_selected_users_sum) * tag_item_count
                    else:
                        raise TypeError("Invalid method `{}`, `method` only support `norm`, `simple` and `tfidf`".format(method))
                else:
                    if method == 'norm':
                        recommend[item_index] += (tag_count / marked_tag_sum) * (tag_item_count / selected_item_sum)
                    elif method == 'simple':
                        recommend[item_index] += tag_count * tag_item_count
                    elif method == 'tfidf':
                        recommend[item_index] += tag_count / np.log(1 + tag_selected_users_sum) * tag_item_count
                    else:
                        raise TypeError("Invalid method `{}`, `method` only support `norm`, `simple` and `tfidf`".format(method))
        # print(time.time() - t1)
        sorted_recommend = sorted(recommend.items(), key=lambda x: (x[1]), reverse=True)[:n]
        return {user_id: dict(sorted_recommend)}

    def __eval(self, train_recommend, test_data):
        user_id = [i for i in train_recommend.keys()][0]
        test_data_item = test_data['bookmarkID'].unique()
        tp = len(set(test_data_item) & set(train_recommend[user_id].keys()))
        # for item_id in test_data_item:
        #     if item_id in train_recommend[user_id]:
        #         tp += 1
        return tp

    def fit(self, train_data):
        self.calc_result = self.__calc_frequency(train_data)

    def predict(self, user_id, n, method='simple'):
        return self.__calc_item_recommendation(user_id,
                                               self.calc_result['user_item'],
                                               self.calc_result['user_tag'],
                                               self.calc_result['tag_item'],
                                               n,
                                               method)

    def eval(self, n, test_data):
        print("Calculating...It might take a while")
        t1 = time.time()
        test_data_user_id = test_data['userID'].unique()
        total_tp = 0
        tpfp = 0
        tpfn = 0
        check = []
        for user_id in test_data_user_id:
            train_recommend = self.predict(user_id, n, method='simple')
            user_test_data = test_data[test_data['userID'] == user_id]
            total_tp += self.__eval(train_recommend, user_test_data)
            tpfn += len(user_test_data['bookmarkID'].unique())
            tpfp += n
            check.append((user_id, total_tp, tpfn, tpfp))
        recall = total_tp / tpfn
        precision = total_tp / tpfp
        t2 = time.time()
        print("Recall: %10.4f" % (recall * 100))
        print("Precision: %10.4f" % (precision * 100))
        print('execute time: ', '%.5f'%(t2 - t1))
        return recall, precision, check

if __name__ == '__main__':
    file_path = "user_taggedbookmarks-timestamps.dat"
    n, ratio = 50, 0.4
    tb = TagBased(file_path, '\t')
    train_test_data = tb.train_test_split(ratio, 88)
    tb.fit(train_test_data['train_data'])
    calc_result = tb.calc_result
    # 使用3种方法，预测用户id为8 的排序前10的item
    p1_simple = tb.predict(8, 10)
    p1_tf = tb.predict(8, 10, method='tfidf')
    p1_normal = tb.predict(8, 10, method='norm')
    recall, precision, check = tb.eval(n, train_test_data['test_data'])

# split 0.2, n = 10, simple
# Recall:     0.0358
# Precision:     0.1103
# execute time:  156.39369  (eval time)

# split 0.4, n = 10, simple
# Recall:     0.0793
# Precision:     0.3482
# execute time:  128.15915

# split 0.4, n = 30, simple
# Recall:     0.1351
# Precision:     0.1977
# execute time:  109.89463

# split 0.3, n = 20, simple
# Recall:     0.0724
# Precision:     0.1397
# execute time:  136.17357

# split0.4, n = 50, simple
# Recall:     0.1661
# Precision:     0.1458
# execute time:  105.10650

# split 0.4, n = 10, norm
# Recall:     0.0694
# Precision:     0.3047
# execute time:  106.90683

# split 0.4, n = 10, tfidf
# Recall:     0.0756
# Precision:     0.3319
# execute time:  193.55620

# split 0.4, n = 50, tfidf
# Recall:     0.1822
# Precision:     0.1600
# execute time:  188.95242