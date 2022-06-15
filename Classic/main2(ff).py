import raw_data_read_test as rd
import label_pre as lp
import cv2
from sklearn.neighbors import KNeighborsClassifier  ###svm
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import tree
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# 原始数据提取
channels = 1  # 图像的通道数，灰度图为1
path_rawdata = r"../face/rawdata"
path_label1 = r"../face/faceDR"
path_label2 = r"../face/faceDS"
label_one_hot, img_name, label_list = lp.label_pre_one_hot(path_label1, path_label2) # 卷标数据提取
img_list = rd.read_rawdata(channels, path_rawdata, img_name)  # 图片矩阵

n = 0
for i in range(len(label_list)):
    num = i - n
    if label_list[num] is None:
        del label_list[num]
        del img_name[num]
        n = n + 1

array_of_img = []
for img in img_list:
    if img is not None:
        img = img / 255.0
        img = cv2.resize(img, (100, 100))
        array_of_img.append(img)
train_images = np.array(array_of_img)
array_of_img = []

array_of_labels = []
for label in label_one_hot:
    if label is not None:
        append_label = label[0]
        array_of_labels.append(int(append_label))
train_labels = np.array(array_of_labels)
array_of_labels = []

# 回调函数
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=3)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
# PCA 图像数据处理
pca = PCA(n_components=85)
newX = pca.fit_transform(X_train)
xx = pca.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# 测试
score = clf.score(X_test, y_test)
print('决策树训练准确率(含维度降维)：', score * 100, '%')
# 定义模型
knn = KNeighborsClassifier(n_neighbors=3)  # knn
svm = sklearn.svm.SVC()  # svm
knn.fit(newX, y_train)
svm.fit(newX, y_train)

# knn预测
y_pred_on_train = knn.predict(xx)
y_pred_on_train_svm = svm.predict(xx)

acc = metrics.accuracy_score(y_test, y_pred_on_train)
acc_svm = metrics.accuracy_score(y_test, y_pred_on_train_svm)
print('knn训练准确度(含维度降维)(K=3)：', acc * 100, '%')
print('svm训练准确度(含维度降维)：', acc_svm * 100, '%')



