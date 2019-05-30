import torch
from torch import nn, optim
from torchvision import models
from copy import deepcopy
from torch.autograd import Variable
from torchsummary import summary

def EuclideanDistance(a,b):
    a = a.view(-1)
    b = b.view(-1)
    s = 0
    for x,y in zip(a,b):
        s += (x-y)**2

    return s**(1.0/2)


class Model(nn.Module):
    def __init__(self,embedding_size):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(1,5,3)
        self.conv2 = nn.Conv2d(5,10,3)
        self.maxpl = nn.MaxPool2d(kernel_size = 2)
        self.linear = nn.Linear(5290,embedding_size)

    def forward(self,input):
        input = input.view(-1,1,50,50)
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpl(x)
        x = x.view(1,-1)
        return self.linear(x)


model = Model(embedding_size = 25)

a = torch.rand((1,50,50))
p = torch.rand((1,50,50))
n = torch.rand((1,50,50))

opt = optim.SGD(model.parameters(),lr=0.01)
margin = 2

zero = torch.Tensor([0])
for e in range(20):
    
    opt.zero_grad()
    a_em , p_em, n_em = model(a), model(p), model(n)
    dis_ap = EuclideanDistance(a_em,p_em)
    dis_an = EuclideanDistance(a_em,n_em)

    loss = dis_ap - dis_an + margin

    if loss < 0:
        continue

    loss.backward()
    opt.step()


print(EuclideanDistance(model(a),model(p)))
print(EuclideanDistance(model(a),model(n)))



