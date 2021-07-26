import torch
import math

class auto_grad_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)
    @staticmethod
    def backward(ctx,grad_before_one):
        input, = ctx.saved_tensors
        return grad_before_one * 1.5 * (5 * input ** 2 - 1)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

x=torch.linspace(-math.pi,math.pi,2000,dtype=dtype,device=device)
y=torch.sin(x)

a=torch.full((), 0.0,device=device,dtype=dtype,requires_grad=True)
b=torch.full((),-1.0,device=device,dtype=dtype,requires_grad=True)
c=torch.full((), 0.0,device=device,dtype=dtype,requires_grad=True)
d=torch.full((), 0.3,device=device,dtype=dtype,requires_grad=True)
print(a,a.dtype)
learn_rate=8e-6

for i in range(2000):
    P3 = auto_grad_func.apply
    y_pred =a + b * P3(c + d * x)
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
P3 = auto_grad_func.apply
with torch.no_grad():
    res=y_pred =a + b * P3(c + d * x)
    print("sin(pi/6):{}".format(res))
