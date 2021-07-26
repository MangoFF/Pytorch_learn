import torch
import torch.nn as nn
import math
import numpy as np

device=torch.device("cpu")
dtype=torch.float32 #the tensor dtype

x=torch.linspace(-math.pi,math.pi,2000,dtype=dtype,device=device).to(device)
y=torch.sin(x).to(device)


XX=torch.tensor([[x_sli,x_sli**2,x_sli**3] for x_sli in x]).to(device)
#p = torch.tensor([1, 2, 3])
#xx = x.unsqueeze(-1).pow(p) #with the same usage
model=nn.Sequential( torch.nn.Linear(3, 1),#the input is [batch_size,3],the out_put is [
    torch.nn.Flatten(0, 1))#specialize the para because default from 1 to -1(batch_size be ignore)
learn_rate=1e-3
loss_func=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.RMSprop(model.parameters(),lr=learn_rate)
for i in range(2000):
    y_pred = model(XX)#y_pred shape is [2000] because of the flatten is from 0 to 1
    loss=loss_func(y,y_pred)
    if(i+1)%10==0:
        print("{},loss:{}".format(i+1,loss))
    optimizer.zero_grad() #no use model.zero_grad() because the freezen part of the model
    loss.backward()
    optimizer.step()


x=torch.tensor([math.pi/6,(math.pi/6)**2,(math.pi/6)**3])
x=x.reshape(-1,3)
print(x.shape)
res=model(x)
print("sin(pi/6):{}".format(res[0]))
