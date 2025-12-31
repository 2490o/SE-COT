import os
import errno

import copy # 记得导入 copy

from tqdm import tqdm
import pickle as pkl
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from pymage_size import get_image_size

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

all_class_name = ['bus' ,'bike', 'car', 'motor', 'person', 'rider' ,'truck']


# # --- 新增辅助函数：用于加载增强数据 ---
# def load_corrupt_dicts(datasets_root, original_sub_folder, split, corrupt_img_dir):
#     """
#     1. 调用 files2dict 读取原始数据集（daytime_clear）的标注
#     2. 修改 file_name 指向增强后的图片文件夹
#     """
#     # 构造原始数据的路径，例如: .../diverseWeather/daytime_clear
#     original_root = os.path.join(datasets_root,  original_sub_folder)
#
#     # 获取原始数据的字典列表 (会自动使用缓存)
#     original_dicts = files2dict(original_root, split)
#
#     # 深拷贝，避免修改了缓存里的原始数据
#     new_dicts = copy.deepcopy(original_dicts)
#
#     print(f"Loading corrupt images from: {corrupt_img_dir}")
#
#     # 遍历修改图片路径
#     for d in new_dicts:
#         # 获取文件名 (例如 00001.jpg)
#         filename = os.path.basename(d["file_name"])
#
#         # 构造新文件名 (cp + 00001.jpg)
#         new_filename = "cp" + filename
#
#         # 指向新的文件夹路径
#         d["file_name"] = os.path.join(corrupt_img_dir, new_filename)
#
#         # 标注信息 (annotations) 保持不变
#
#     return new_dicts


def get_annotation(root, image_id, ind):
    annotation_file = os.path.join(root,'VOC2007', "Annotations", "%s.xml" % image_id)
    if os.path.getsize(annotation_file) == 0:
        print(f"[EMPTY XML] {annotation_file}")
    et = ET.parse(annotation_file)
    objects = et.findall("object")                                              
    
    record = {}
    record["file_name"] = os.path.join(root, 'VOC2007', "JPEGImages", "%s.jpg" % image_id)
    img_format = get_image_size(record["file_name"])
    w, h = img_format.get_dimensions()

    record["image_id"] = image_id#ind for pascal evaluation actual image name is needed 
    record["annotations"] = []

    for obj in objects:
        class_name = obj.find('name').text.lower().strip()
        if class_name not in all_class_name:
            print(class_name)
            continue
        if obj.find('pose') is None:
            obj.append(ET.Element('pose'))
            obj.find('pose').text = '0'

        if obj.find('truncated') is None:
            obj.append(ET.Element('truncated'))
            obj.find('truncated').text = '0'

        if obj.find('difficult') is None:
            obj.append(ET.Element('difficult'))
            obj.find('difficult').text = '0'

        bbox = obj.find('bndbox')
        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = max(0,float(bbox.find('xmin').text) - 1) # fixing when -1 in anno
        y1 = max(0,float(bbox.find('ymin').text) - 1) # fixing when -1 in anno
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        box = [x1, y1, x2, y2]
        
        #pascal voc evaluator requires int 
        bbox.find('xmin').text = str(int(x1))
        bbox.find('ymin').text = str(int(y1))
        bbox.find('xmax').text = str(int(x2))
        bbox.find('ymax').text = str(int(y2))


        record_obj = {
        "bbox": box,
        "bbox_mode": BoxMode.XYXY_ABS,
        "category_id": all_class_name.index(class_name),
        }
        record["annotations"].append(record_obj)

    if len(record["annotations"]):
        #to convert float to int
        et.write(annotation_file)
        record["height"] = h
        record["width"] = w
        return record

    else:
        return None

def files2dict(root,split):

    cache_dir = os.path.join(root, 'cache')

    pkl_filename = os.path.basename(root)+f'_{split}.pkl'
    pkl_path = os.path.join(cache_dir,pkl_filename)

    if os.path.exists(pkl_path):
        with open(pkl_path,'rb') as f:
            return pkl.load(f)
    else:
        try:
            os.makedirs(cache_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(e)
            pass    

    dataset_dicts = []
    image_sets_file = os.path.join( root,'VOC2007', "ImageSets", "Main", "%s.txt" % split)

    with open(image_sets_file) as f:
        count = 0

        for line in tqdm(f):
            record = get_annotation(root,line.rstrip(),count)
 
            if record is not None:
                dataset_dicts.append(record)
                count +=1 

    with open(pkl_path, 'wb') as f:
        pkl.dump(dataset_dicts,f)
    return dataset_dicts


def register_dataset(datasets_root):
    datasets_root = os.path.join(datasets_root,'diverseWeather')
    dataset_list = ['daytime_clear', 
                    'daytime_foggy',
                    'night_sunny',
                    'night_rainy',
                    'dusk_rainy',
                    ]
    settype = ['train','test']
    
    for name in dataset_list:
        for ind, d in enumerate(settype):
        
                DatasetCatalog.register(name+"_" + d, lambda datasets_root=datasets_root,name=name,d=d \
                    : files2dict(os.path.join(datasets_root,name), d))
                MetadataCatalog.get(name+ "_" + d).set(thing_classes=all_class_name,evaluator_type='pascal_voc')
                MetadataCatalog.get(name+ "_" + d).set(dirname=datasets_root+f'/{name}/VOC2007')
                MetadataCatalog.get(name+ "_" + d).set(split=d)
                MetadataCatalog.get(name+ "_" + d).set(year=2007)
    # # 2. === 新增：注册你的增强数据集 ===
    #
    # # 设定增强数据集的名字
    # aug_name = "daytime_clear_train_cp"
    #
    # # 设定增强图片的绝对路径 (请确保这个路径是正确的！)
    # # 假设你的 cp_train_corrupt_img 在 datasets_root 下，如果不是请填绝对路径
    # aug_img_path = os.path.join(datasets_root, 'daytime_clear', 'VOC2007', 'cp_train_corrupt_img')
    #
    # DatasetCatalog.register(aug_name, lambda: load_corrupt_dicts(
    #     datasets_root,
    #     "daytime_clear",  # 复用 daytime_clear 的标注
    #     "train",  # 复用 train 的 split
    #     aug_img_path  # 图片换成这里的
    # ))
    #
    # # 注册元数据 (类别等信息与原始数据一致)
    # MetadataCatalog.get(aug_name).set(thing_classes=all_class_name, evaluator_type='pascal_voc')
    #
    # print(f"Registered {aug_name} using images from {aug_img_path}")