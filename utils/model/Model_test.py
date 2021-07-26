import torch.cuda
import cv2
from CNN import  CNN_model
import torchvision
import numpy as np
def test_CNN(img_path,class_name):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_model(3).to(device)
    model.load_state_dict(torch.load('./weight/CNN.ckpt'))
    with torch.no_grad():
        img = cv2.resize(cv2.imread(img_path), (300, 300)).astype(np.float32) / 255
        img = img.reshape(-1, 3, 300, 300)
        img = torch.from_numpy(img).to(device)
        print(img.shape)
        model.eval()
        output = model(img)
        print(output[0])
        print(class_name[output[0].argmax(0)])
def test_Resnet_finetune(img_path,class_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = torchvision.models.resnet18(pretrained=True).to(device)
    resnet.load_state_dict(torch.load("./weight/ResnetFinetune.ckpt"))
    with torch.no_grad():
        img = cv2.resize(cv2.imread(img_path), (300, 300)).astype(np.float32) / 255
        img = img.reshape(-1, 3, 300, 300)
        img = torch.from_numpy(img).to(device)
        print(img.shape)
        resnet.eval()
        output = resnet(img)
        print(output[0])
        print(class_name[output[0].argmax(0)])
if __name__=="__main__":
    test_CNN("C:\\Users\\93911\\Desktop\\ClassifyDemo\\example\\2.jpg",["paper","rock","scissors"])