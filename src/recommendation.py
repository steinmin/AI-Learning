import random
import math
import operator
import pandas as pd

from os.path import dirname, join
current_dir = dirname(__file__)

records = {}

# 训练集，测试集
train_data = dict()
test_data = dict()

# 用户标签，商品标签
user_tags = dict()
tag_items = dict()
user_items = dict()
tag_users = dict()

# 数据加载
def load_data(path):
    print("开始数据加载...")
    df = pd.read_csv(join(current_dir, path), sep='\t')
    for i in range(len(df)):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tag = df['tagID'][i]
        # 键不存在时，设置默认值{}
        records.setdefault(uid,{})
        records[uid].setdefault(iid,[])
        records[uid][iid].append(tag)
    print("数据集大小为 %d." % (len(df)))
    print("设置tag的人数 %d." % (len(records)))
    print("数据加载完成\n")

    # 将数据集拆分为训练集和测试集
def train_test_split(ratio, seed=100):
    random.seed(seed)
    for u in records.keys():
        for i in records[u].keys():
            # ratio比例设置为测试集
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

    # remove the users who are in test_data, but not in train_data
    for key in [k for k in test_data.keys() if k not in train_data]:
        test_data.pop(key, None)
    print("训练集样本数 %d, 测试集样本数 %d" % (len(train_data),len(test_data)))


# 使用dictionary统计值的个数,每有重复，值加1。
def count_weight(count_dict, key, value, weight=1):
    if key not in count_dict:
        count_dict.setdefault(key, {})
        count_dict[key].setdefault(value, weight)
    else:
        if value not in count_dict[key]:
            count_dict[key][value] = weight
        else:
            count_dict[key][value] += weight


# 使用训练集，初始化user_tags, tag_items, user_items, tag_users
def initStat():
    for user, bookmarks in train_data.items():
        for bookmark,tags in bookmarks.items():
            for tag in tags:
                #print tag
                # 用户和tag的关系
                count_weight(user_tags, user, tag)
                # tag和bookmark的关系
                count_weight(tag_items, tag, bookmark)
                # 用户和bookmark的关系
                count_weight(user_items, user, bookmark)
                # tag和用户的关系
                count_weight(tag_users, tag, user)
    print("user_tags, tag_items, user_items初始化完成.")
    print("user_tags: %d   user_items: %d" % (len(user_tags), len(user_items)))
    print("tag_users: %d,   tag_items: %d" % (len(tag_users), len(tag_items)))


# 对用户user推荐Top-N
def recommend_simple_tag(user, N):
    simpleTag=dict()
    # 对Item进行打分，分数为所有的（用户对某标签使用的次数 wut, 乘以 商品被打上相同标签的次数 wti）之和
    for tag, wut in user_tags[user].items(): # 用户使用标签t的次数
        for item, wti in tag_items[tag].items(): #商品i被打过标签t的次数
            if item in user_items[user]: continue
            # 用户对该商品打过标签，就对该商品的SimpleTag进行累加
            if item not in simpleTag: simpleTag[item] = 0
            simpleTag[item] += wut * wti
    return sorted(simpleTag.items(), key=operator.itemgetter(1), reverse=True)[0:N]

# 对用户user推荐Top-N
def recommend_norm_tag(user, N):
    simpleTag=dict()
    # 对Item进行打分，分数为所有的（用户对某标签使用的次数 wut, 乘以 商品被打上相同标签的次数 wti）之和
    for tag, wut in user_tags[user].items(): # 用户使用标签t的次数
        for item, wti in tag_items[tag].items(): #商品i被打过标签t的次数
            if item in user_items[user]: continue

            #norm
            norm = len(user_tags[user].items()) * len(tag_items[tag].items())
            # 用户对该商品打过标签，就对该商品的SimpleTag进行累加
            if item not in simpleTag: simpleTag[item] = 0
            simpleTag[item] += wut * wti / norm
    return sorted(simpleTag.items(), key=operator.itemgetter(1), reverse=True)[0:N]

# 对用户user推荐Top-N
def recommend_tfidf_tag(user, N):
    simpleTag=dict()
    # 对Item进行打分，分数为所有的（用户对某标签使用的次数 wut, 乘以 商品被打上相同标签的次数 wti）之和
    for tag, wut in user_tags[user].items(): # 用户使用标签t的次数
        for item, wti in tag_items[tag].items(): #商品i被打过标签t的次数
            if item in user_items[user]: continue

            #norm
            norm = math.log(1 + len(tag_users[tag].items()))
            # 用户对该商品打过标签，就对该商品的SimpleTag进行累加
            if item not in simpleTag: simpleTag[item] = 0
            simpleTag[item] += wut * wti / norm
    return sorted(simpleTag.items(), key=operator.itemgetter(1), reverse=True)[0:N]


# 使用测试集，计算准确率和召回率, 两率与Tag没有关系，只和商品有关系
def precisionAndRecall(N, recommend):
    hit, h_recall, h_precision = (0, 0, 0)
    for user,items in test_data.items():
        # 获取Top-N推荐列表
        rank = recommend(user, N)
        for item, _ in rank:
            if item in items:   # TP：打过标签的被推荐， FN：打过标签的没被推荐， FP:没打过标签的被推荐
                hit += 1        #相当于TP： 推荐的商品在用户给过标签的商品列表里，算命中一次
        h_recall += len(items)  #TP + FN, 打过标签的商品总数
        h_precision += N        #TP + FP， 推荐的商品总数
    #print('一共命中 %d 个, 一共推荐 %d 个, 用户设置tag总数 %d 个' %(hit, h_precision, h_recall))
    # 返回准确率 和 召回率
    return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))