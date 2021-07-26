import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
class Yolo_dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        lines= open(annotations_file,"r").readlines()
        img_name = []
        boxes = []
        for j, (item) in enumerate(lines):
            item = item.rstrip('\n')
            item = item.split(" ")
            img_name.append(item[0])
            boxes.append([])
            for i in range(1, len(item)):
                box = item[i].split(",")
                box = list(map(int, box))
                boxes[j].append(box)
        self.imgs=[]
        for i in range(len(self.name_list)):
            img_path = os.path.join(self.img_dir, self.name_list[i])
            image = cv2.imread(img_path)
            self.imgs.append(image)
        self.name_list=img_name
        self.box_list=boxes
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.box_list)

    def __getitem__(self, idx):
        label = self.box_list[idx]
        if self.transform:
            self.imgs[idx] = self.transform(self.imgs[idx])
        if self.target_transform:
            label = self.target_transform(label)
        return self.imgs[idx], label


if __name__=="__main__":
    #Yolo_dataSet
    data_train=Yolo_dataset("C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\Yolo\\train\\_annotations.txt",
                                 "C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\Yolo\\train\\",)
    while True:
        randn=torch.randint(len(data_train),size=(1,)).item()
        img,boxes=data_train[randn]
        for box in boxes:
            cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color=(0,0,255))
        cv2.imshow("mango",img)
        cv2.waitKey(1111)


