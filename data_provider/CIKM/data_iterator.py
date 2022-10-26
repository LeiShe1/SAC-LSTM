import sys
import os


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from core.utils.util import *
from torch.utils import data
import imageio
# from scipy.misc import imread
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
imsave = imageio.imsave
imread = imageio.imread
import codecs

"这个类是将图片的转换成torch格式，分成输入5张和输出5张"
class CIKM_Datasets(data.Dataset):
    def __init__(self,root_path):
        self.root_path = root_path

    def __getitem__(self, index):
        self.folds = self.root_path+'sample_'+str(index+1)+'/'
        files = os.listdir(self.folds)
        files.sort()   #对列表进行排序 这里有点没对
        imgs = []
        for file in files:
            imgs.append(imread(self.folds+file)[:,:,np.newaxis])
        imgs = np.stack(imgs,0)
        imgs = torch.from_numpy(imgs).cuda()
        in_imgs = imgs[:5]
        out_imgs = imgs[5:]
        return in_imgs,out_imgs

    def __len__(self):
        return len(os.listdir(self.root_path))


class Norm(object):
    def __init__(self, max=255):
        self.max = max

    def __call__(self, sample):
        video_x = sample
        # new_video_x = (video_x / self.max - 0.5 ) * 2
        new_video_x = video_x / self.max  
        return new_video_x


class ToTensor(object):

    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


class CIKM_Datasets_custom(data.Dataset):
    def __init__(self, configs, data_train_path, data_test_path, mode, transform=None):
        self.mode = mode
        self.configs = configs
        self.transform = transform
        if self.mode == "train":
            print("Loading train dataset")
            self.path = data_train_path
            with codecs.open(self.path) as f:
                self.file_list = f.readlines()
                # print(self.file_list)
            print("Loading train dataset finished, with size", len(self.file_list))
        else:
            print('Loading test dataset')
            self.path = data_test_path
            with codecs.open(self.path) as f:
                self.file_list = f.readlines()
            print('Loading test dataset finished, with size:', len(self.file_list))
                
    def __getitem__(self,idx):
        item_ifo_list = self.file_list[idx].split(',')
        # print(item_ifo_list)
        begin = int(item_ifo_list[1])
        end = 16
        data_slice = np.ndarray(shape=(end - begin, 101,  101, 1), dtype=np.uint8)
        for i in range(end - begin):
            file_index = i + 1 
            file_name = "img_" + str(file_index) + ".png"
            image = cv2.imread(str(item_ifo_list[0]) + file_name,  cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
            # image = image[1:257, 1:257]
            data_slice[i, :] = image
        # print(str(item_ifo_list[0]))
        sample = data_slice
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.file_list)




def data_process(filename,data_type,dim=None,start_point = 0):
    save_root = '/mnt/A/CIKM2017/CIKM_datasets/'+data_type+'/'
    if start_point == 0:
        clean_fold(save_root)

    with open(filename) as fr:
        if data_type == 'train':
            sample_num = 10000
            validation = random.sample(range(1, 10000 + 1), 2000)
            save_validation_root = '/mnt/A/CIKM2017/CIKM_datasets/validation/'
            clean_fold(save_validation_root)
        elif data_type == 'test':
            sample_num = 2000+start_point
        print('the number of '+data_type+' datasets is:',str(sample_num))
        validation_count = 1
        train_count = 1
        for i in range(start_point+1,sample_num+1):
            print(data_type+' data loading complete '+str(100.0*(i+1)/sample_num)+'%')
            if data_type == 'train':
                if i in validation:
                    save_fold = save_validation_root+'sample_'+str(validation_count)+'/'
                    validation_count = validation_count + 1
                else:
                    save_fold = save_root + 'sample_' + str(train_count) + '/'
                    train_count = train_count + 1
            else:
                save_fold = save_root+'sample_'+str(i)+'/'
            clean_fold(save_fold)

            line = fr.readline().strip().split(' ')
            cate = line[0].split(',')
            id_label = [cate[0]]
            record = [int(cate[2])]
            length = len(line)

            for i in range(1, length):
                record.append(int(line[i]))

            mat = np.array(record).reshape(15, 4, 101, 101).astype(np.uint8)

            # deals with -1
            mat[mat == -1] =  0

            if dim == None:
                pass
            else:
                mat = mat[:,dim]

            for t in range(1,16):
                img = mat[t-1]
                # print(img.shape)
                img_name = 'img_'+str(t)+'.png'
                imsave(save_fold+img_name,img)


def sub_sample(batch_size,mode = 'random',data_type='train',index = None,type = 7):
    if type not in [4,5,6,7]:
        raise ('error')
    save_root = '/mnt/A/CIKM2017/CIKM_datasets/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,8000)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range((16-type-8),16):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imread(img_path)[:,:,np.newaxis]

                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range((16-type-8), 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>4001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1    #b_cup : 3
        for batch_idx in range(batch_size):
            if index>4001:  # index : 1
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range((16-type-8), 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==4001:
            return imgs,(index,0)
        return imgs,(index,b_cup)

    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>2001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range((16-type-8), 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")

import cv2

def sample(batch_size,mode = 'random',data_type='train',index = None):
    save_root = '/data/data-home/shelei/data/CIKM_Radar_Data/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,8000)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range(1,16):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imread(img_path)[:,:,np.newaxis]

                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index == 8001:
                    return imgs,(index,0)
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    print(img_path)
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    #dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)   index=1
    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>4001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1    #3
        for batch_idx in range(batch_size): #进行4次循环
            if index>4001:
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis] #第三个维度上增加，为1
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1

        imgs = np.array(imgs)
        if index==4001:
            return imgs,(index,0)
        return imgs,(index,b_cup)   #返回： 图片，（4，3）

    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>2001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index==2001:
                return imgs,(index,0)
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")


def sample_test(batch_size,mode = 'random',data_type='test_1023_modif',index = None):
    save_root = '/data/data-home/shelei/data/CIKM_Radar_Data/' + data_type + '/'
    #dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)   index=1
    if index == None:
        raise('index need be initialize')
    if index>4001 or index<1:
        raise('index exceed')
    imgs = []
    b_cup = batch_size-1    #3
    for batch_idx in range(batch_size): #进行4次循环
        if index == 1025:
            return imgs,(index,0)
        if index>4001:
            index = 4001
            b_cup = batch_idx
            imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
            break
        img_fold = save_root + 'sample_'+str(index)+'/'
        batch_imgs = []
        for t in range(1, 16):
            img_path = img_fold + 'img_' + str(t) + '.png'
            img = imread(img_path)[:, :, np.newaxis] #第三个维度上增加，为1
            batch_imgs.append(img)
        imgs.append(np.array(batch_imgs))
        index = index+1

    imgs = np.array(imgs)
    if index==4001:
        return imgs,(index,0)
    return imgs,(index,b_cup)   #返回： 图片，（4，3）





from skimage.transform import resize

def sample_Rard(batch_size,mode = 'random',data_type='train',index = None):
    save_root = '/home/shelei/self_data/Radar_CR_TrainDataset_pre_data' + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                # sample_index = random.randint(1,8000)
                # img_fold = save_root + 'sample_'+str(sample_index)+'/'
                list_fold = os.listdir(save_root)
                choice_num = random.choice(list_fold)
                img_fold = save_root + str(choice_num) + '/'
                batch_imgs = []
                for t in range(0,15):
                    img_path = img_fold + str(choice_num) + '_' +str(t)+'.png'
                    img = imread(img_path)[:,:,np.newaxis]
                    img = resize(img, (128, 128))
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    #dat, (index, b_cup) = sample(batch_size, data_type='test', index=index)   index=1
    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>4001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1    #3
        for batch_idx in range(batch_size): #进行4次循环
            if index>4001:
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis] #第三个维度上增加，为1
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1

        imgs = np.array(imgs)
        if index==4001:
            return imgs,(index,0)
        return imgs,(index,b_cup)   #返回： 图片，（4，3）

    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>2001 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")

if __name__ == '__main__':
    pass
