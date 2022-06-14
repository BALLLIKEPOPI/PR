import torch.cuda
import csv
import pandas as pd
import raw_data_read_test as rd
import label_pre as lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
import random

# 注释即跑GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 训练集
channels = 1  # 图像的通道数，灰度图为1
path_rawdata = r"../face/rawdata"
path_label = r"../face/faceDR"
label_one_hot, img_name = lp.label_pre_one_hot(path_label)
img_list = rd.read_rawdata(channels, path_rawdata, img_name) # 图片矩阵

# 测试集
test_path_rawdata = r"../face/rawdata"
test_path_label = r"../face/faceDS"
test_label_one_hot, test_img_name = lp.label_pre_one_hot(test_path_label)
test_img_list = rd.read_rawdata(channels, test_path_rawdata, test_img_name) # 图片矩阵

# for img in img_list:
#     if img is not None:
#         cv2.imshow("asdf", img)
#         cv2.waitKey(1)

label_list, img_name = lp.read_label("../face/faceDR")
label_name, label_encoded_list = lp.encode_label(label_list)
prop_num = len(label_name[-1])
label_list = lp.one_hot_prop(label_encoded_list, prop_num)
label_one_hot = lp.one_hot_label(label_list, label_name)
print(label_name)
print(label_list)

test_label_list, test_img_name = lp.read_label("../face/faceDS")
test_label_name, test_label_encoded_list = lp.encode_label(test_label_list)
prop_num = len(test_label_name[-1])
test_label_list = lp.one_hot_prop(test_label_encoded_list, prop_num)
test_label_one_hot = lp.one_hot_label(test_label_list, test_label_name)

for i in range(1400):
    img_list.append(test_img_list[0])
    label_list.append(test_label_list[0])
    img_name.append(test_img_name[0])
    del test_img_name[0]
    del test_img_list[0]
    del test_label_list[0]

#######################################################################

n = 0
for i in range(len(label_list)):
    num = i - n
    if label_list[num] is None:
        del label_list[num]
        # del img_name[num]
        n = n+1

array_of_img = []
for img in img_list:
    if img is not None:
        img = img / 255.0
        img = cv2.resize(img, (100, 100))
        array_of_img.append(img)
train_images = np.array(array_of_img)
array_of_img = []

#######################################################################

n = 0
for i in range(len(test_label_list)):
    num = i - n
    if test_label_list[num] is None:
        del test_label_list[num]
        # del test_img_name[num]
        n = n+1

array_of_img = []
for img in test_img_list:
    if img is not None:
        img = img / 255.0
        img = cv2.resize(img, (100, 100))
        array_of_img.append(img)
test_images = np.array(array_of_img)
array_of_img = []

#######################################################################

array_of_labels = []
for label in label_list:
    if label is not None:
        append_label = label[0]
        array_of_labels.append(int(append_label))
train_labels = np.array(array_of_labels)
array_of_labels = []

#######################################################################

array_of_labels = []
for label in test_label_list:
    if label is not None:
        append_label = label[0]
        array_of_labels.append(int(append_label))
test_labels = np.array(array_of_labels)
array_of_labels = []

# sample = random.randint(0, 2000)
# plt.figure()
# plt.imshow(test_images[sample][:, ::-1], cmap='gray')
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    sample = random.randint(0, 2000)
    plt.imshow(train_images[sample][:, ::-1], cmap='gray')
    index = int(label_list[sample][0])
    plt.xlabel(label_name[0][index])
plt.show()

# 高斯噪声
data_augmentation = keras.Sequential(
    [
        keras.layers.GaussianNoise(0.1, input_shape=(100, 100, 1)),
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# plt.figure(figsize=(10, 10))
# for i in range(9):
#     sample = random.randint(0, 500)
#     temp = train_images[sample:sample+1]
#     augmented_images = data_augmentation(temp)
#     ax = plt.subplot(3, 3, i + 1)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(augmented_images[0][:, ::-1], cmap='gray')
# plt.show()

# 设置层
model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(25, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(50, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(100, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(200, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
model.summary()

# train_images = np.expand_dims(train_images, axis=2)
# train_images = np.expand_dims(train_images, axis=3)

# 回调函数
X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=3)
# del train_images
# del train_labels
save_weights = 'save_weights.h5'
last_weights = 'last_weights.h5'
best_weights = 'best_weights.h5'
# model.load_weights(best_weights)

checkpoint = keras.callbacks.ModelCheckpoint(best_weights, monitor='val_accuracy', save_best_only=True, mode='max',
                                             verbose=1)
reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=0, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
callbacks = [checkpoint]

# hist = model.fit(train_images, train_labels, epochs=2000)
hist = model.fit(X_train, Y_train, epochs=500, validation_data=(X_val, Y_val), use_multiprocessing=True,
                 callbacks=callbacks, workers=3)

model.save_weights(last_weights)
plt.figure()

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')  # 'bo'为画蓝色圆点，不连线
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation accuracy,Training and validation loss')
plt.legend()  # 绘制图例，默认在右上角

plt.show()

##########################################################################
#####                            预测                                 #####
##########################################################################
model.load_weights(best_weights) # 加载之前已保存的能复现结果的最佳权重
# model.load_weights(save_weights) # 加载之前已保存的能复现结果的最佳权重
del train_images
del train_labels

predictions = model.predict(test_images)
results = np.argmax(predictions, axis=1)
submissions = pd.read_csv('test.csv')
submissions['label'] = results
submissions.to_csv('submission.csv', index=False)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    sample = random.randint(0, 599)
    plt.imshow(test_images[sample][:, ::-1],  cmap='gray')
    plt.xlabel(label_name[0][results[sample]])
plt.show()

filename = "submission.csv"
list1 = []
with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    column = [row['label'] for row in reader]

# 数据处理
for i in range(len(test_label_list)):
    if test_label_list[i][0] == 1:
        test_label_list[i][0] = 0
    else:
        test_label_list[i][0] = 1

for i in range(599):
    column[i] = int(column[i])

# 计算准确率
correct = 0
error = 0
for i in range(599):
    if test_label_list[i][0] == column[i]:
        correct = correct + 1
    if test_label_list[i][0] != column[i]:
        error = error + 1
print("number of correct:")
print(correct)
print("number of error:")
print(error)
print("accuracy:")
print((correct/599))
