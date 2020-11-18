#!/usr/bin/env python
# coding: utf-8

# # 名企作业 2020-11-21, 石敏

# ## Action 2

# ## 对Titanic数据进行清洗，使用之前介绍过的10种模型中的至少2种（包括TPOT）
# 数据集地址：https://github.com/cystanford/Titanic_Data
#
# 数据集中的字段描述： PassengerId 乘客编号 Survived 是否幸存 Pclass 船票等级 Name 乘客姓名 Sex 乘客性别 SibSp 亲戚数量（兄妹、配偶数） Parch 亲戚数量（父母、子女数） Ticket 船票号码 Fare 船票价格 Cabin 船舱 Embarked 登录港口



# 由上述数据可以得知需要预测的是“是否能幸存”
# 查空值， 查异常值， Label_Encode for Sex, Embarked
#
# - survival - Survival (0 = No; 1 = Yes)
# - class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - name - Name
# - sex - Sex
# - age - Age
# - sibsp - Number of Siblings/Spouses Aboard
# - parch - Number of Parents/Children Aboard
# - ticket - Ticket Number
# - fare - Passenger Fare
# - cabin - Cabin
# - embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# - boat - Lifeboat (if survived)
# - body - Body number (if did not survive and body was recovered)

# MLFrame.py 是根据核心课上学习的内容，我自己改写的一个不同AI算法的调用框架
import MLFrame as mlf


ml = mlf.MLearning()

ml.read_train('./titanic/train.csv')
ml.read_test('./titanic/test.csv')

# Merge train and test
ml.add_label_for_test('Survived')
ml.merge_train_and_test()

# 数据清洗, Age , Cabin, Embarked 有空值
ml.fill_none('Age', ml.data['Age'].mean())
ml.fill_none('Fare', ml.data['Fare'].mean())
ml.fill_none('Embarked', ml.data['Embarked'].value_counts().index[0])

# Cabin的值缺失了超过77%： （1308 - 295）/ 1308， 所以补齐并没有实际意义
# 票号不统一，可以试一试，也许有一线可能性，比如编号小的是头等舱，生还机率大。
# 但在此练习中，Fare应该能体现船舱级别， 因此扔掉此列
# 名字，没法转成数字，也不会对结果产生什么影响，所以也去掉
ml.drop_column(['Ticket', 'Cabin', 'Name'])

# Label-Encode
ml.label_encoder(['Sex', 'Embarked'])
# Min_Min_Scaler
ml.min_max_scaler(['Fare'])
# One-Hot
ml.one_hot(['Pclass', 'Sex', 'Embarked'])

# 特征选择
ml.set_features_without(['PassengerId','Survived'])

# Logistic Regression 逻辑回归
ml.predict('lr')
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_lr.csv')


# ID3 决策树
ml.predict('id3_c')
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_id3_c.csv')

# AdaBoost
ml.predict('ada_c')
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_ada_c.csv')


# XGBoost
pred_lr = ml.predict('xgb_c')
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_xgb_c.csv')


#  TPOT
#ml.run_tpot('./tpot_mnist_pipeline.py')

# Result from TPOT
'''
Generation 1 - Current best internal CV score: 0.8105677419354839
Generation 2 - Current best internal CV score: 0.8153419354838709
Generation 3 - Current best internal CV score: 0.8154064516129031
Generation 4 - Current best internal CV score: 0.8154064516129031
Generation 5 - Current best internal CV score: 0.8249935483870969
Best pipeline: XGBClassifier(input_matrix, learning_rate=0.1, max_depth=7, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.8500000000000001)
0.8283582089552238

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8249935483870969
exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=7, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.8500000000000001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
'''

modelSetup = mlf.ModelSetup()
modelSetup.set_values('xgb_c', 'learning_rate=0.1, max_depth=7, min_child_weight=6, n_estimators=100, nthread=1, subsample=0.8500000000000001')

ml.predict('xgb_c', modelSetup)
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_xgb_c_tpot.csv')
