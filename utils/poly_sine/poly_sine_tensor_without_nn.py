import torch
import math
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

x=torch.linspace(-math.pi,math.pi,2000,dtype=dtype,device=device)
y=torch.sin(x)

a=torch.randn((),device=device,dtype=dtype)
b=torch.randn((),device=device,dtype=dtype)
c=torch.randn((),device=device,dtype=dtype)
d=torch.randn((),device=device,dtype=dtype)
print(a,a.dtype)
learn_rate=1e-6

for i in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss=torch.square(y-y_pred).sum().item()
    if(i+1)%100==0:
        print("{},loss:{}".format(i,loss))
    grad_y_pred=2*(y_pred-y)
    grad_a=(grad_y_pred*1).sum()*learn_rate
    grad_b=(grad_y_pred*x).sum()*learn_rate
    grad_c=(grad_y_pred*x**2).sum()*learn_rate
    grad_d=(grad_y_pred*x**3).sum()*learn_rate
    a-=grad_a
    b-=grad_b
    c-=grad_c
    d-=grad_d
x=math.pi/6
res=a + b * x + c * x ** 2 + d * x ** 3
print("sin(pi/6):{}".format(res))
