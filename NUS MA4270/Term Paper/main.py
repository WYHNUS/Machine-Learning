import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def draw_confusion_matrices(cms, classes):
    fig = plt.figure(figsize = (10, 15))
    i = 1   # used to compute the matrix location
    for clf_name, cm in cms.items():
        thresh = cm.max() / 2   # used for the text color
        
        ax = fig.add_subplot(len(cms) / 2 + 1, 2, i,
                             title = 'Confusion matrix for %s' % clf_name, 
                             xlabel = 'Predicted',
                             ylabel = 'True')
        cax = ax.matshow(cm, cmap = plt.cm.Blues)
        fig.colorbar(cax)
        i += 1
        
        # Ticks
        ax.set_xticklabels([''] + classes)
        ax.set_yticklabels([''] + classes)
        ax.tick_params(labelbottom = True, labelleft = True, labeltop = False)
        
        # Text
        for x in range(len(cm)):
            for y in range(len(cm[0])):
                ax.text(y, x, cm[x, y], 
                        horizontalalignment = 'center', 
                        color = 'black' if cm[x, y] < thresh else 'white')
        
    plt.tight_layout()
    plt.show()

X_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train = y_train.values.ravel()

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_test_lst = y_test['match'].values.tolist()

# try to normalize...
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

classifiers = {
    "RBF SVM": svm.SVC(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators = 100)
}

matrices = {}
for clf_name, clf in classifiers.items():
	start = time.time()
	trained_clf = clf.fit(X_train, y_train)
	Y_predict = trained_clf.predict(X_test)

	misclassified_label_count = 0
	for i in range(len(Y_predict)):
		if (int(y_test_lst[i]) != int(Y_predict[i])):
			misclassified_label_count += 1

	end = time.time()
	t = end - start
	print ('accuracy for ' + clf_name + ' is ' + str(1 - (1.0 * misclassified_label_count / len(Y_predict))) + ' with running time ' + str(t))

	# draw confusion matrix
	matrices[clf_name] = confusion_matrix(y_test_lst, Y_predict)

labels = np.unique(y_test_lst).tolist()
draw_confusion_matrices(matrices, labels)