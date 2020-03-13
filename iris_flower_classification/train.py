"""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv("iris.data", names=names)

array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Testing various models
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
for name,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state=1, shuffle=True)
    cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_result)
    print(f'{name}: {cv_result.mean():.2} {cv_result.std():.2}')
    
"""

# Evaluate predictions

"""
print(accuracy_score(Y_test, predict))
print(confusion_matrix(Y_test, predict))
print(classification_report(Y_test, predict))
"""
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predict = model.predict(X_test)

pickle.dump(model, open('iris_flower_classifier','wb'))


