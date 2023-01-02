from Decision_Tree_45 import Decision_Tree_C45
from Decision_Tree_gini import Decision_Tree_C45_gini
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:\pythonprogram\machine_learning\dataset\Client_Info.csv",encoding='unicode_escape')
data = np.array(data)
feat_names = ['x1', 'x2', 'x3', 'x4', 'x5']
train, test = train_test_split(data, test_size=0.3, random_state=42)
model = Decision_Tree_C45()
tree = model.create_decision_tree(train, feat_names, depth=-1, minsample=200)
# print(tree)
pred_labels = []
for i in range(len(test)):
    label = model.predict(tree, feat_names, test[i])
    pred_labels.append(label)
acc = 0
for i in range(len(test)):
    if pred_labels[i] == test[i, -1]:
        acc += 1.0
print("Decision_Tree_C45 accuracy:%.3f" % (acc / len(test)))
model = Decision_Tree_C45_gini()
tree = model.create_decision_tree(train, feat_names, depth=-1, minsample=200)
# print(tree)
pred_labels = []
for i in range(len(test)):
    label = model.predict(tree, feat_names, test[i])
    pred_labels.append(label)
acc = 0
for i in range(len(test)):
    if pred_labels[i] == test[i, -1]:
        acc += 1.0
print("Decision_Tree_C45_gini accuracy:%.3f" % (acc / len(test)))
from sklearn.tree import DecisionTreeClassifier

# Add code here to build decision tree model using sklearn
model_skl = DecisionTreeClassifier(criterion="entropy")
model_skl.fit(train[:, 0:-1], train[:, -1])
pred_labels_skl = model_skl.predict(test[:, 0:-1])
acc = 0
for i in range(len(test)):
    if pred_labels_skl[i] == test[i, -1]:
        acc += 1.0
print("Decision_Tree_SKL accuracy:%.3f" % (acc / len(test)))
