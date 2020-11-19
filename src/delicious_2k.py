# ## Action 1

# ## 针对Delicious数据集，对SimpleTagBased算法进行改进（使用NormTagBased、TagBased-TFIDF算法）
# Delicious数据集：https://grouplens.org/datasets/hetrec-2011/
#
# 1867名用户，105000个书签，53388个标签
#
# 格式：
# userID           bookmarkID            tagID                timestamp

import recommendation as rec


file_path = "./delicious_2k/user_taggedbookmarks-timestamps.dat"
# 字典类型，保存了user对item的tag，即{userid: {item1:[tag1, tag2], ...}}

rec.load_data(file_path)

# 训练集，测试集拆分，20%测试集
rec.train_test_split(0.2)

rec.initStat()

print("推荐结果评估(Simple Tag Based)：")
print("%3s %10s %10s" % ('N',"精确率",'召回率'))
for n in [5,10,20,40,60,80,100]:
    precision,recall = rec.precisionAndRecall(n, rec.recommend_simple_tag)
    print("%3d %10.3f%% %10.3f%%" % (n, precision * 100, recall * 100))


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
