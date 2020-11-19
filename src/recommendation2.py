import random
import math
import operator
import pandas as pd
import numpy as np

from os.path import dirname, join
current_dir = dirname(__file__)

class RecommendAlgr(object):
    def __init__(self):
        # 训练集，测试集
        self.records = dict()
        self.test_data = dict()

        # 用户标签，商品标签
        self.user_tags = dict()
        self.tag_items = dict()
        self.user_items = dict()
        self.tag_users = dict()

    def __addValueToMat(self, mat, index, item, value=1):
        if index not in mat:
            mat.setdefault(index,{})
            mat[index].setdefault(item,value)
        else:
            if item not in mat[index]:
                mat[index][item] = value
            else:
                mat[index][item] += value

    def add_record(self, user, item, tag):
        self.records.setdefault(user, {})
        self.records[user].setdefault(item, [])
        self.records[user][item].append(tag)


    def train_test_split(self, ratio=0.2, random_state=40):
        np.random.seed(random_state)
        for user in self.records.keys():
            for item in self.records[user].keys():
                if random.random()<ratio: # ratio比例设置为测试集
                    self.test_data.setdefault(user, {})
                    self.test_data[user].setdefault(item, [])
                    self.test_data[user][item] = self.records[user][item]
                else:
                    for tag in self.records[user][item]:
                        self.__addValueToMat(self.user_tags,  user, tag)
                        self.__addValueToMat(self.tag_items,  tag,  item)
                        self.__addValueToMat(self.user_items, user, item)
                        self.__addValueToMat(self.tag_users,  tag,  user)

        self.records = None ## records 已经不需要了
        # remove the users who are in test_data, but not in train_data
        usersInTrain = self.user_items.keys()
        for user in [user for user in self.test_data.keys() if user not in usersInTrain]:
            self.test_data.pop(user, None)

        print("训练集样本数 %d, 测试集样本数 %d" % (len(usersInTrain),len(self.test_data)))

    # 对用户user推荐Top-N
    def recommend(self, user, N, algr):
        simpleTag=dict()
        # 对Item进行打分，分数为所有的（用户对某标签使用的次数 wut, 乘以 商品被打上相同标签的次数 wti）之和
        for tag, wut in self.user_tags[user].items(): # 用户使用标签t的次数
            for item, wti in self.tag_items[tag].items(): #商品i被打过标签t的次数
                if item in self.user_items[user]: continue
                norm = 1
                if algr == 'norm_tag_based':
                    norm = len(self.user_tags[user].items()) * len(self.tag_items[tag].items())
                elif algr == 'tag_based_tfidf':
                    norm = math.log(1 + len(self.tag_users[tag].items()))
                simpleTag.setdefault(item, 0)
                simpleTag[item] += wut * wti / norm
        return sorted(simpleTag.items(), key=operator.itemgetter(1), reverse=True)[0:N]

    # 使用测试集，计算准确率和召回率, 两率与Tag没有关系，只和商品有关系
    def precisionAndRecall(self, N, algr='simple_tag_based'):
        hit, h_recall, h_precision = (0, 0, 0)
        for user,items in self.test_data.items():
            # 获取Top-N推荐列表
            rank = self.recommend(user, N, algr)
            for item, _ in rank:
                if item in items:   # TP：打过标签的被推荐， FN：打过标签的没被推荐， FP:没打过标签的被推荐
                    hit += 1        #相当于TP： 推荐的商品在用户给过标签的商品列表里，算命中一次
            h_recall += len(items)  #TP + FN, 打过标签的商品总数
            h_precision += N        #TP + FP， 推荐的商品总数
        # 返回准确率 和 召回率
        return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))


# 使用RecommendAlgr计算用户标签推荐
file_path = "./delicious_2k/user_taggedbookmarks-timestamps.dat"
rec = RecommendAlgr()

df = pd.read_csv(join(current_dir, file_path), sep='\t')
for i in range(len(df)):
    rec.add_record(df['userID'][i], df['bookmarkID'][i], df['tagID'][i])

rec.train_test_split()



print("推荐结果评估(Simple Tag Based)：")
print("%3s %10s %10s" % ('N',"精确率",'召回率'))
for n in [5]: #,10,20,40,60,80,100]:
    precision,recall = rec.precisionAndRecall(n, 'simple_tag_based')
    print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))

"""
print("推荐结果评估(Norm Tag Based)：")
print("%3s %10s %10s" % ('N',"精确率",'召回率'))
for n in [5,10,20,40,60,80,100]:
    precision,recall = rec.precisionAndRecall(n, rec.recommend_norm_tag)
    print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))


print("推荐结果评估(Tag Based - TFIDF)：")
print("%3s %10s %10s" % ('N',"精确率",'召回率'))
for n in [5,10,20,40,60,80,100]:
    precision,recall = rec.precisionAndRecall(n, rec.recommend_tfidf_tag)
    print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))

    """