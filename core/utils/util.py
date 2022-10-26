
import numpy as np
import shutil
import copy
import os


def nor(frames):#归一化
    new_frames = frames.astype(np.float32)/255.0
    return new_frames

def de_nor(frames):#将归一化后的像素返回到255
    new_frames = copy.deepcopy(frames)
    new_frames *= 255.0
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def normalization(frames,up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up/2)
    new_frames -= 1
    return new_frames

def denormalization(frames,up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames

def clean_fold(path):  #如果存在文件，则先删除后再创建
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)