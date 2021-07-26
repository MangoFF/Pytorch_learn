import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
class File_classify_dataset(Dataset):
    device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    def __init__(self, img_dirs,size=(30,30), transform=None, target_transform=None):
        self.img_dirs=img_dirs
        self.size=size
        self.className=[]
        self.img_name_lists=[]
        self.label_lists = []
        self.classNum=0
        self.transform = transform
        self.target_transform = target_transform
        self.img_lists = []
        class_id = 0
        for root, dirs, files in os.walk(img_dirs):
            if (len(dirs) > 0):
                self.classNum = len(dirs)
                self.className = self.className+dirs
            else:
                self.img_name_lists = self.img_name_lists + files
                self.label_lists = self.label_lists + [class_id] * len(files)
                class_id += 1
        self.label_lists=torch.tensor(self.label_lists).to(File_classify_dataset.device)
        for i in range(len(self.label_lists)):
            img_path = os.path.join(self.img_dirs, self.className[self.label_lists[i]], str(self.img_name_lists[i]))
            # 归一化和标准化都要在dataset里面做
            image = torch.from_numpy(cv2.resize(cv2.imread(img_path), self.size).astype(np.float32) / 255).to(File_classify_dataset.device)
            self.img_lists.append(image)

    def __len__(self):
        return len(self.label_lists)

    def __getitem__(self, idx):
        label = self.label_lists[idx]
        if self.transform:
            self.img_lists[idx] = self.transform(self.img_lists[idx])
        if self.target_transform:
            label = self.target_transform(label)
        return self.img_lists[idx], label

if __name__ == "__main__":
    # File_Classify
    data_train = File_classify_dataset("C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\File_Classify",(100,100))
    while True:
        randn = torch.randint(len(data_train), size=(1,)).item()
        img, label = data_train[randn]
        img=(img.to(torch.device("cpu")).numpy()*255).astype(np.uint8)
        cv2.imshow(str(label), img)
        cv2.waitKey(111)
