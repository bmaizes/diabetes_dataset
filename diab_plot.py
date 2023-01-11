from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

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
# print(sum(train))
X_train = X[train]
y_train = y[train]

for feature in range(7,8):
    plt.scatter(X_train[:,6], X_train[:,feature], c=y_train, cmap='jet_r')
    plt.xlabel('Feature 6')
    plt.ylabel('Feature {:.2f}'.format(feature))        
    plt.show()

