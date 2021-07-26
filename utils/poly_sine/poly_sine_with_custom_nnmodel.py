import torch
import torch.nn as nn
import math
import numpy as np
class custom_model(nn.Module):
    def __init__(self):
        super(custom_model,self).__init__()
        self.a=torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
    def forward(self,x):
        return self.a+self.b*x+self.c*x**2+self.d*x**3
    def string(self):
        return "mango"
device=torch.device("cpu")
dtype=torch.float32 #the tensor dtype

x=torch.linspace(-math.pi,math.pi,2000,dtype=dtype,device=device).to(device)
y=torch.sin(x).to(device)

model=custom_model()

learn_rate=1e-6
loss_func=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=learn_rate)
for i in range(2000):
    y_pred = model(x)#y_pred shape is [2000] because of the flatten is from 0 to 1
    loss=loss_func(y,y_pred)
    if(i+1)%10==0:
        print("{},loss:{}".format(i+1,loss))
    optimizer.zero_grad() #no use model.zero_grad() because the freezen part of the model
    loss.backward()
    optimizer.step()


x=torch.tensor([math.pi/6])
x=x.reshape(-1,1)
print(x.shape)
with torch.no_grad():
    res=model(x)
    print("sin(pi/6):{}".format(res[0]))
