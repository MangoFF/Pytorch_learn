import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset.File_classify import File_classify_dataset
from torch.utils.data import DataLoader

# Hyper parameters
num_epochs = 5
num_classes = 3
batch_size = 64
learning_rate = 0.001

class CNN_model(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN_model,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Linear(87616, num_classes)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        res=self.fc(out)
        return res
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = File_classify_dataset("C:\\Users\\93911\\Desktop\\ClassifyDemo\\data\\File_Classify",
                                          size=(300, 300))
    train_load = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model=CNN_model(3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step=len(train_load)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_load):
            images=images.reshape(-1,3,300,300)
            out = model(images)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epch:{},{}/{},loss:{:.4f}".format(epoch,i + 1, total_step, loss.item()))

    torch.save(model.state_dict(),'./weight/CNN.ckpt')