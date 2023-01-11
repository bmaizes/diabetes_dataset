from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# load the dataset
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

# print(predict([[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], [0, 0, 0, 0, 0, 0, 0, 0]]))

y_pred = predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(accuracy_score(y_train, y_pred))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(accuracy_score(y_train, y_pred))

logist = LogisticRegression()
logist.fit(X_train, y_train)
y_pred = logist.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(accuracy_score(y_train, y_pred))

sc = StandardScaler()
lda = LDA(n_components=1)
lda.fit(X_train, y_train)

print('Logistic Regr Coef:', logist.coef_)
print('Logistic Regr Bias:', logist.intercept_)
print("LDA Coef:", lda.coef_)

test = np.array([232,235,337,284,171,189,637,105,307,123,287,505,80,463,606,479,481,86,4,445,24,187,559,448,51,474,653,197,359,83,544,122,252,645,16,362
,121,367,681,693,686,635,583,71,585,188,363,255,594,279,85,382,626,64
,394,160,643,28,10,178,11,289,728,708,230,719,37,669,258,438,504,624
,323,76,294,75,500,588,336,236,205,119,501,627,146,370,652,676,371,144
,162,565,308,84,199,358,342,644,277,513,633,286,605,34,18,727,39,697
,525,614,173,374,216,91,322,599,361,198,698,383,574,183,292,257,615,319
,578,618,638,447,431,437,49,523,715,220,128,240,67,730,639,159,724,365
,632,575,675,127,709,631,149,0,668,377,60,124,538,95,657,176,410,59
,733,537,99,265,100,248,346,491,442,590,629,623,625,114,427,515,514,7
,66,300,666,702,388,487,608,411,600,541,104,470,426,604,163,177,654,68
,729,558,458,276,275,345,678,226,38,8,143,372,403,658,209,55,557,206
,589,73,716,586,440,98,50,714,130,285,720,113,368,713,90,340,341,663
,329,115,222,349,26,169,533,597,444,63,690,744,291,233,227,111])
X_test = X[test]
y_test = y[test]

y_pred = predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

y_pred = logist.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))



validation = np.array([135,356,674,204,420,467,621,263,726,425,646,231,423,718,381,408,670,344,507,78,47,696,136,126,35,88,293,390,269,468,404,552,711,324,486,446,154,620,539,740,343,435,732,524,182,664,496,392,566,736,380,691,254,746
,519,239,165,418,547,41,651,535,622,102,747,274,36,553,134,455,587,234
,563,311,354,17,682,642,229,43,548,107,695,210,567,33,679,490,112,531
,616,506,221,299,454,387,103,738,129,655,215,603,46,320,44,673,327,32
,268,6,259,465,570,497,576,270,184,92,391,330,133,484,110,560,710,318
,534,27,452,369,561,393,407,672,712,262,582,717,397,510,692,138,325,360
,151,453,568,591,348,494,742,735,364,564,412,436,352,540,613,57,20,61
,14,242,332,15,309,333,522,571,53,168,301,271,536,680,244,555,432,131
,224,498,706,200,207,601,530,152,520,223,593,161,428,213,647,253,687,278
,662,93,703,281,58,249,389,175,302,31,378,592,77,243,245,689,485,472
,495,464,473,303,297,1,499,451,665,155,489,518,179,74,634,267,471,321
,147,72,366,526,94,562,460,118,247,684,13,581,283,290,335,282])
X_valid = X[validation]
y_valid = y[validation]

