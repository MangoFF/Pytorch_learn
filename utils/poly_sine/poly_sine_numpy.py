import numpy as np
import math
epoches=2000
batch_size=2000

if __name__=="__main__":
    #the source input
    x=np.linspace(-math.pi,math.pi,batch_size)
    y=np.sin(x)

    #random the weight
    a=np.random.randn()
    b=np.random.randn()
    c=np.random.randn()
    d=np.random.randn()
    print("{},{},{},{}".format(a,b,c,d))
    loss_aver=0
    for i in range(epoches):
        pre_y=a+b*x+c*x**2+d*x**3
        loss=np.square(y-pre_y).sum()
        loss_aver=loss_aver+loss
        if (i+1)%10==0:
            print("{},loss:{:.1f}".format(i,loss_aver/10))
            loss_aver=0
        learn_rate= 1e-6
        a-=2*(pre_y-y).sum()*learn_rate
        b-=2*((pre_y-y)*x).sum()*learn_rate
        c-=2*((pre_y-y)*x**2).sum()*learn_rate
        d-=2*((pre_y-y)*x**3).sum()*learn_rate
    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
    x=math.pi/6
    res=a+b*x+c*x**2+d*x**3
    print(f"result of sin(pai/6){res}")



