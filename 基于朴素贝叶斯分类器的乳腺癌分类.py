from sklearn.datasets import load_breast_cancer
from Naive_Bayes_Classifier import NaiveBayesClassifier
cancer_data = load_breast_cancer()
data = cancer_data['data']#feature space
y = cancer_data['target']#label space
feature_names = cancer_data['feature_names']
label_names = [0,1]# 1 for malignant, 0 for benign
feature_types = [0 for i in range(len(feature_names))]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=0.3)
model = NaiveBayesClassifier()
model.fit(X_train, y_train, feature_names, feature_types, label_names)
pred_test = model.predict(X_test, feature_names, feature_types)
acc = 0.0
for i in range(len(y_test)):
    if y_test[i] == pred_test[i]:
        acc += 1.0
print("NaiveBayesClassifier Accuracy:%.3f"%(acc/len(y_test)))
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_test = gnb.predict(X_test)
acc = 0.0
for i in range(len(y_test)):
    if y_test[i] == pred_test[i]:
        acc += 1.0
print("sklearn GaussianNB Accuracy:%.3f"%(acc/len(y_test)))
