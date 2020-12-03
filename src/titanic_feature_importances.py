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
ml.predict('ctb')
ml.save_predict('PassengerId', 'Survived', './titanic/baseline_lr.csv')

print(ml.feature_importances)