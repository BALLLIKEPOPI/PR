#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/12 12:12
# @Author  : Chen HanJie
# @FileName: label_pre.py
# @Software: PyCharm

import re


def encode_label(label_list: list) -> (list, list):
    label_name = [[] for i in range(5)]

    label_encoded_list = []

    for label in label_list:
        if label is None:
            label_encoded_list.append(None)
            continue
        label_encoded = []

        for i in range(len(label) - 1):
            if not label[i] in label_name[i]:
                label_name[i].append(label[i])
                label_encoded.append(len(label_name[i]) - 1)
            else:
                label_encoded.append(label_name[i].index(label[i]))

        if label[4] is None:
            label_encoded.append([None])
        else:
            prop = []
            for ll in label[4]:
                if not ll in label_name[4]:
                    label_name[4].append(ll)
                    prop.append(len(label_name[4]) - 1)
                else:
                    prop.append(label_name[4].index(ll))
            label_encoded.append(prop)

        label_encoded_list.append(label_encoded)
    return label_name, label_encoded_list


def read_label(path: str) -> (list, list):
    label_list = []
    img_name = []
    with open(path, "r") as f:
        r = f.readline()
        while r:
            img_name.append(r[:5].strip())
            t = re.search('_missing descriptor', r)
            if t is not None:
                label_list.append(None)
                r = f.readline()
                continue
            else:
                label1 = []
                pattern = re.compile(r'\((.*?)\)')
                t = pattern.findall(r)
                for s in t[0:-1]:
                    c = s.split(" ")[-1].strip()
                    label1.append(c)

                pattern = re.compile(r'\'\((.*?)\)')
                t = pattern.findall(r)[0].strip()
                if not t == '':
                    prop = []
                    t = t.split(" ")
                    for i in t:
                        prop.append(i)
                    label1.append(prop)
                else:
                    label1.append(None)
            label_list.append(label1)
            r = f.readline()
    return label_list, img_name


def one_hot_prop(label_encoded_list: list, prop_num: int) -> list:
    label_list = []
    iii = 0
    for label in label_encoded_list:
        iii += 1
        if label is not None:
            prop_one_hot = [0 for i in range(prop_num)]
            prop = label.pop()
            for i in prop:
                if i is not None:
                    prop_one_hot[i] = 1
            label.append(prop_one_hot)
            label_list.append(label)
            if 1 in prop_one_hot:
                pass
        else:
            label_list.append(None)
    return label_list


def one_hot(index, num):
    label_one_hot = [0 for i in range(num)]
    label_one_hot[index] = 1
    return label_one_hot


def one_hot_label(label_list: list, label_name) -> list:
    label_one_hot = []
    for label in label_list:
        if label is not None:
            l = []
            for i in range(len(label_name) - 1):
                o = one_hot(label[i], len(label_name[i]))
                l += o
            l += label[-1]
        else:
            label_one_hot.append(None)
            continue
        label_one_hot.append(l)
    return label_one_hot


def label_pre_one_hot(path):
    label_list, img_name = read_label(path)
    label_name, label_encoded_list = encode_label(label_list)
    prop_num = len(label_name[-1])
    label_list = one_hot_prop(label_encoded_list, prop_num)
    label_one_hot = one_hot_label(label_list, label_name)

    return label_one_hot, img_name


if __name__ == '__main__':
    label_list, img_name = read_label("../face/faceDR")
    sum1 = 0
    n = 0
    # funny 24个删除
    for i in range(len(label_list)):
        num = i - n
        if label_list[num] is not None:
            if label_list[num][0] == 'male':
                if label_list[num][3] == 'funny':
                    print(label_list[num])
                    del label_list[num]
                    del img_name[num]
                    n = n + 1
    sum1 = n
    n = 0

    #  200个smiling
    for i in range(len(label_list)):
        num = i - n
        if label_list[num] is not None:
            if label_list[num][0] == 'male':
                if label_list[num][3] == 'smiling':
                    print(label_list[num])
                    del label_list[num]
                    del img_name[num]
                    n = n + 1
                    if n == 350:
                        break
    #  200个serious
    sum1 += n
    n = 0
    for i in range(len(label_list)):
        num = i - n
        if label_list[num] is not None:
            if label_list[num][0] == 'male':
                if label_list[num][3] == 'serious':
                    print(label_list[num])
                    del label_list[num]
                    del img_name[num]
                    n = n + 1
                    if n == 350:
                        break
    #  2个black
    sum1 +=  n
    n = 0
    for i in range(len(label_list)):
        num = i - n
        if label_list[num] is not None:
            if label_list[num][0] == 'male':
                if label_list[num][2] == 'black':
                    print(label_list[num])
                    del label_list[num]
                    del img_name[num]
                    n = n + 1
                    if n == 30:
                        break
    #  200个white
    sum1 += n
    n = 0
    for i in range(len(label_list)):
        num = i - n
        if label_list[num] is not None:
            if label_list[num][0] == 'male':
                if label_list[num][2] == 'white':
                    print(label_list[num])
                    del label_list[num]
                    del img_name[num]
                    n = n + 1
                    if n == 30:
                        break
    print(sum1)

    label_name, label_encoded_list = encode_label(label_list)
    prop_num = len(label_name[-1])
    label_list = one_hot_prop(label_encoded_list, prop_num)
    label_one_hot = one_hot_label(label_list, label_name)
