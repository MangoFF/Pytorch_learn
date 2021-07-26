import torch
import math
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

x=torch.linspace(-math.pi,math.pi,2000,dtype=dtype,device=device)
y=torch.sin(x)

a=torch.randn((),device=device,dtype=dtype,requires_grad=True)
b=torch.randn((),device=device,dtype=dtype,requires_grad=True)
c=torch.randn((),device=device,dtype=dtype,requires_grad=True)
d=torch.randn((),device=device,dtype=dtype,requires_grad=True)
print(a,a.dtype)
learn_rate=1e-6

for i in range(2000):
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    loss=torch.square(y-y_pred).sum()
    if(i+1)%100==0:
        print("{},loss:{}".format(i,loss))
    loss.backward()
    with torch.no_grad():
        #manual optim
        a -= a.grad * learn_rate
        b -= b.grad * learn_rate
        c -= c.grad * learn_rate
        d -= d.grad * learn_rate
        #manual zero grad
        a.grad=None
        b.grad = None
        c.grad = None
        d.grad = None
x=math.pi/6
res=a + b * x + c * x ** 2 + d * x ** 3
print("sin(pi/6):{}".format(res))
