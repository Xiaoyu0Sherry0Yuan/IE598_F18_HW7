import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',header=None)
df.head()
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=30, stratify=y)
for g in [1,50,100,500,1500]:
    forest = RandomForestClassifier(n_estimators=g,random_state=1)
    forest.fit(X_train, y_train)
    scores_train = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    print('n_estimators=', str(g))
    print('CV accuracy scores of training samples: %s' % scores_train)
    print('mean of scores of training samples: %.3f'% np.mean(scores_train))
    print('standard deviation of scores of training samples: %.3f'% np.std(scores_train),'\n')


feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

print("My name is Xiaoyu Yuan")
print("My NetID is: 664377413")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
