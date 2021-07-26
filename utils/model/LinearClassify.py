import torch
from dataset.File_classify import File_classify_dataset
import torch.nn as nn
import cv2
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=30*30*3
hidden_size=50
num_classes=3
num_epochs=2
learning_rate=0.001
batch_size=100
annotation_path="C:\\Users\\93911\\PycharmProjects\\ClassifyDemo\data\\train\\_annotations.txt"
pictur_path="C:\\Users\\93911\\PycharmProjects\\ClassifyDemo\data\\train\\"

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        x= self.flatten(x)
        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        return out

if __name__=="__main__":
    # 模型评价：
    # 训练出来的模型文件500mb，这都比yolo的训练文件还要大了，但是效果却非常的差，所以这种模型需要优化

    #dataset and data loader
    train_dataset=File_classify_dataset("C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\File_Classify")
    train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    className=train_dataset.className
    #build model
    model=NeuralNet(input_size,hidden_size,num_classes).to(device)

    #criterion and optimizer
    criterion=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #train the model
    total_step=len(train_dataset)//batch_size
    for epoch in range(num_epochs):
        for i,(imgs,labels) in enumerate(train_loader):
            #reshape to [batchsize,channel,data]
            imgs=imgs.reshape(-1,3,30*30,).to(device)
            labels=labels.to(device)

            #model use and calc loss
            outputs=model(imgs)
            loss=criterion(outputs,labels)

            #backward the grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(i+1)%10==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    # Save the model checkpoint
    torch.save(model.state_dict(), 'LinearClassify.ckpt')
    with torch.no_grad():
        img_origin=cv2.resize(cv2.imread("../../2.jpg"),(30,30)).astype(np.float32)/255.
        img=img_origin.reshape(-1,3,30*30)
        img=torch.tensor(img).to(device)
        print(img.shape)
        imgs=model(img)
        print(className[imgs[0].argmax(0)] )



