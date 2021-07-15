# coding=utf-8
import os
import os.path as osp
import json
import cv2
import glob
import numpy as np
import re
import collections
import mmcv
import shutil
def get_labels_json_info():
    annotations = []
    images = []
    obj_count = 0
    text_map = open(r'D:\code\multiblur\13633212419907022849\mapfile.txt', "r")
    tweets = []
    for line in open(r'D:\code\multiblur\13633212419907022849/labelfile.json', "r", encoding='utf-8'):
        tweets.append(json.loads(line))
    length=len(tweets)
    img_dir = 'D:/raw_data/'
    save_dir_train = 'D:/code/multiblur/rect/train/'
    if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)
    for idx, j in enumerate(tweets[:int(0.9*length)]):
        try:
            pth = j["path"]
            line = text_map.readline()
            line = line.replace('\n', '').replace('\t', '')
            labelid, img_pth = line[:20], line[20:]
            # print(labelid, img_pth)
            assert labelid == pth
            img_id = img_pth.split('/')[-1]
            filename = img_id
            images.append(dict(
                id=idx,
                file_name=filename,
                height=300,
                width=300))
            #dir_pth = img_pth.rstrip(img_id)
            img_pth_r = img_dir + img_pth
            #img = cv2.imread(img_pth)
            shapes = j["Tags"]["_n8dYXoBfCgrj7Me0nAj"]["shapes"]
            shutil.copy(img_pth_r, save_dir_train+filename)
            for shape in shapes:
                # print(shape["tag"])
                lbl_blur = shape["tag"]["模糊程度"]["label_value_id"]  # 1001,1002...
                x, y = [pt["x"] for pt in shape['pts']], [pt["y"] for pt in shape['pts']]
                x_min, y_min, x_max, y_max = (
                    min(x), min(y), max(x), max(y))
                polygon_points = list(zip(x, y))
                poly = [p for x in polygon_points for p in x]
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=lbl_blur,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1
            print("train:{}".format(idx))
        except Exception as e:
            print("ERROR:{}".format(e))
    coco_format_json_train = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 1001, 'name': 'blur'}, {'id': 1002, 'name': 'bclur'}, {'id': 1003, 'name': 'clear'}, {'id': 1004, 'name': 'cclear'}])
    mmcv.dump(coco_format_json_train, r'D:\code\multiblur\coco_rect_singleblur_train.json')
    annotations = []
    images = []
    obj_count = 0
    save_dir_test = 'D:/code/multiblur/rect/test/'
    if not os.path.exists(save_dir_test):
        os.makedirs(save_dir_test)
    for idx, j in enumerate(tweets[int(0.9*length):]):
        try:
            pth = j["path"]
            line = text_map.readline()
            line = line.replace('\n', '').replace('\t', '')
            labelid, img_pth = line[:20], line[20:]
            # print(labelid, img_pth)
            assert labelid == pth
            img_id = img_pth.split('/')[-1]
            filename = img_id
            images.append(dict(
                id=idx,
                file_name=filename,
                height=300,
                width=300))
            #dir_pth = img_pth.rstrip(img_id)
            img_pth_r = img_dir + img_pth
            #img = cv2.imread(img_pth)
            shapes = j["Tags"]["_n8dYXoBfCgrj7Me0nAj"]["shapes"]
            shutil.copy(img_pth_r, save_dir_test + filename)
            for shape in shapes:
                # print(shape["tag"])
                lbl_blur = shape["tag"]["模糊程度"]["label_value_id"]  # 1001,1002...
                x, y = [pt["x"] for pt in shape['pts']], [pt["y"] for pt in shape['pts']]
                x_min, y_min, x_max, y_max = (
                    min(x), min(y), max(x), max(y))
                polygon_points = list(zip(x, y))
                poly = [p for x in polygon_points for p in x]
                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=lbl_blur,
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    area=(x_max - x_min) * (y_max - y_min),
                    segmentation=[poly],
                    iscrowd=0)
                annotations.append(data_anno)
                obj_count += 1
            print("test:{}".format(idx))
        except Exception as e:
            print("ERROR:{}".format(e))
    coco_format_json_test = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 1001, 'name': 'blur'}, {'id': 1002, 'name': 'bclur'}, {'id': 1003, 'name': 'clear'},
                    {'id': 1004, 'name': 'cclear'}])
    mmcv.dump(coco_format_json_test, r'D:\code\multiblur\coco_rect_singleblur_test.json')

def ti2custom():
    text_map = open(r'D:\code\multiblur\13633212419907022849\mapfile.txt', "r")
    tweets = []
    for line in open(r'D:\code\multiblur\13633212419907022849/labelfile.json', "r", encoding='utf-8'):
        tweets.append(json.loads(line))
    print(len(tweets))
    data_infos = []
    k = 0
    for idx, j in enumerate(tweets):
        try:
            pth = j["path"]
            line = text_map.readline()
            line = line.replace('\n', '').replace('\t', '')
            labelid, img_pth = line[:20], line[20:]
            # print(labelid, img_pth)
            assert labelid == pth
            img_id = img_pth.split('/')[-1]
            bboxes = []
            labels = []
            # dir_pth = img_pth.rstrip(img_id)
            # img_pth = img_dir + img_pth
            # img = cv2.imread(img_pth)
            shapes = j["Tags"]["_n8dYXoBfCgrj7Me0nAj"]["shapes"]
            for shape in shapes:
                # print(shape["tag"])
                lbl_blur = shape["tag"]["模糊程度"]["label_value_id"]  # 1001,1002...
                x, y = [pt["x"] for pt in shape['pts']], [pt["y"] for pt in shape['pts']]

                polygon_points = list(zip(x, y))
                pt1, pt3 = polygon_points[0], polygon_points[2]
                pt1, pt3 = list(map(int, list(pt1))), list(map(int, list(pt3)))
                bbox = pt1[0], pt1[1], pt3[0] - pt1[0], pt3[1] - pt1[1]  # x,y,w,h
                bboxes.append([float(b) for b in bbox])
                labels.append(int(lbl_blur))
                # print(polygon_points)
                # img = cv2.rectangle(img, (pt1[0], pt1[1]), (pt3[0], pt3[1]), (0, 255, 0), 2)
                # img = cv2.putText(img, polygon_lbl, (pt1[0], pt1[1]), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255),
                #                 thickness=1)
                # makedirIfNotExit('./0709multibox_big/' + dir_pth + '/' + polygon_lbl)
                # cv2.imwrite('./0709multibox_big/' + dir_pth + '/' + polygon_lbl + '/' + img_id, img)
            data_infos.append(
                dict(
                    filename=img_pth,
                    width=300,
                    height=300,
                    ann=dict(
                        bboxes=bboxes,
                        labels=labels)
                ))
            k += 1
            print(k)
        except Exception as e:
            print("ERROR:{}".format(e))
    with open(r'D:\code\multiblur\middle_rect_singleblur.json', "w", encoding='utf-8') as f:
        json.dump(data_infos, f)
if __name__ == '__main__':
    get_labels_json_info()