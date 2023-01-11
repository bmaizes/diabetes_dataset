from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from collections import Counter


dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

train = np.array([551,419,462,373,596,23,456,584,246,595,457,685,656,723,19,225,69,316,545,219,237,477,400,153,194,317,195,469,203,554,677,353,700,116,492,357
,705,725,602,166,310,355,569,339,482,139,158,402,150,416,120,108,641,737
,439,326,478,280,398,40,699,328,109,3,609,529,577,145,185,156,181,296
,191,572,52,375,434,636,542,22,89,516,521,739,483,9,517,459,148,2
,81,532,512,441,56,424,461,30,256,648,748,5,443,180,527,29,650,137
,660,125,743,409,385,508,433,82,157,25,264,406,261,640,511,630,429,598
,745,488,106,704,164,101,331,313,87,417,196,190,580,97,298,334,617,384
,214,376,208,607,734,549,228,212,466,659,350,493,556,694,295,610,612,405
,314,688,211,193,415,132,140,172,503,683,628,721,386,476,399,117,550,543
,70,528,186,502,422,671,201,288,573,749,305,351,449,218,192,250,347,65
,79,272,96,48,170,238,142,217,379,45,480,509,266,395,42,315,413,202
,174,304,21,414,619,661,430,579,251,241,421,260,731,450,396,722,338,401
,54,12,62,167,546,306,649,475,707,667,701,741,273,141,611,312])
X_train = X[train]
y_train = y[train]

X_copy = X_train
y_copy = y_train

"""Zeroes are reds. Ones are blues"""

def feature_split(feature):
    count = 0
    l = []
    # print(y_train[np.argsort(X_train[:,feature])])
    # print(np.sort(X_train[:,feature]))
    for value in y_copy[np.argsort(X_copy[:,feature])]:
        value = int(value)
        if value == 1:
            count = count + 1
        else:
            count = count - 88/162
        l.append(count)
    # plt.plot(l)
    # plt.show()
    return np.sort(X_copy[:,feature])[l.index(min(l))]

def split_classifier(input):
    vote = 0
    for feature in range(0, 8):
        if input[feature] < feature_split(feature): #its probably a zero (aka a red)
            vote -= 1
        else: #its probably a one (aka a blue)
            vote += 1
    if vote > 0:
        return 1.
    else:
        return 0.

def predict(Xs):
    list = []
    for input in Xs:
        list.append(split_classifier(input))
    return list


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# print(scaler.transform([[3, 84, 68, 30, 106,  31.9, 0.591,  25   ]]))
# print(X_train[0])
# print(X_train_scaled[0])

def start_nn():
    model = Sequential() 
    model.add(Dense(3, input_dim=8, activation='relu', kernel_regularizer=l2(), bias_regularizer=l2()))
    model.add(Dropout(0.35))
    model.add(Dense(3, activation='relu', kernel_regularizer=l2(), bias_regularizer=l2()))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=200, batch_size=10)
    return model

neural = start_nn()
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)
logist = LogisticRegression()
logist.fit(X_train_scaled, y_train)


def benchmark():
    y_pred = predict(X_train)
    print(confusion_matrix(y_train, y_pred))
    print(accuracy_score(y_train, y_pred))

    y_pred = knn.predict(X_train_scaled)
    print(confusion_matrix(y_train, y_pred))
    print(accuracy_score(y_train, y_pred))

    y_pred = logist.predict(X_train_scaled)
    print(confusion_matrix(y_train, y_pred))
    print(accuracy_score(y_train, y_pred))

    y_pred = neural.predict(X_train_scaled)
    rounded = [round(x[0]) for x in y_pred]
    print(confusion_matrix(y_train, rounded))
    print(accuracy_score(y_train, rounded))

# benchmark()

def vote(Xs): #i know it was silly to design the method like this but I wanted to be really clear about the elements
    votes = []
    y_pred_manual = predict(Xs)
    Xs = scaler.transform(Xs)
    y_pred_knn = knn.predict(Xs)
    y_pred_logist = logist.predict(Xs)
    y_pred_nn = neural.predict(Xs)
    y_pred_nn = [round(x[0]) for x in y_pred_nn]
    votes.append(y_pred_manual)
    votes.append(y_pred_knn)
    votes.append(y_pred_nn)
    votes.append(y_pred_logist)
    votes = [int(x[0]) for x in votes]
    occurence_count = Counter(votes)
    return occurence_count.most_common(1)[0][0]

# print(vote([[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]]))
# print(vote([[0, 0, 0, 0, 0, 0, 0, 0]]))

def aggregate(Xs):
    list = []
    for input in Xs:
        list.append(vote([input]))
    return list

y_pred = aggregate(X_copy)
print(confusion_matrix(y_train, y_pred))
print(accuracy_score(y_train, y_pred))


