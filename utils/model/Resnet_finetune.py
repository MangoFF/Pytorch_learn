import torch
from torch.utils.data import DataLoader
import torchvision.models
import torch.nn as nn
from dataset.File_classify import File_classify_dataset
import cv2
import numpy as np
import os
batch_size=64
class_num=3
class_name=[]
epochs=15
learning_rate=0.001


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train_dataset=File_classify_dataset("C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\File_Classify",size=(300,300))
    train_load=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    class_name=train_dataset.className

    resnet = torchvision.models.resnet18(pretrained=True).to(device)
    # If you want to finetune only the top layer of the model, set as below.
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, 3).to(device)
    if(os.path.exists("./weight/ResnetFinetune.ckpt")):
        resnet.load_state_dict(torch.load("./weight/ResnetFinetune.ckpt"))
        # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_load)
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_load):
            imgs = imgs.reshape(-1, 3, 300, 300).to(device)
            labels = labels.to(device)
            outputs = resnet(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
    # Save the model checkpoint
    torch.save(resnet.state_dict(), './weight/ResnetFinetune.ckpt')







