from model import NeuralNet
import torch
import cv2
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=300*300*3
hidden_size=500
num_classes=3
num_epochs=10
learning_rate=0.001
batch_size=100
className=['paper', 'rock', 'scissors']
if __name__=="__main__":
    model = NeuralNet(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load('finger_model.ckpt'))
    with torch.no_grad():
        img_origin=cv2.imread("1.jpg").astype(np.float32)/255
        img_origin=cv2.resize(img_origin,(300,300))
        img=img_origin.reshape(-1,3,300*300)
        img=torch.tensor(img)
        print(img.shape)
        imgs=model(img)
        print(className[imgs[0].argmax(0)] )