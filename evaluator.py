import math
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import ece
import argparse

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getx(self):
        return self.x

    def gety(self):
        return self.y


def GetCross(p1, p2, p):
    return (p2.x - p1.x) * (p.y - p1.y) - (p.x - p1.x) * (p2.y - p1.y)

class Getlen:
  def __init__(self,p1,p2):
    self.x=p1.getx()-p2.getx()
    self.y=p1.gety()-p2.gety()
    #use math.sqrt() to get the square root 用math.sqrt（）求平方根
    self.len= math.sqrt((self.x**2)+(self.y**2))
  #define the function of getting the length of line 定义得到直线长度的函数
  def getlen(self):
    return self.len

def IsPointInMatrix(p1, p2, p3, p4, p):
    isPointIn = GetCross(p1, p2, p) * GetCross(p3, p4, p) >= 0 and GetCross(p2, p3, p) * GetCross(p4, p1, p) >= 0
    return isPointIn

def getDis(p1, p2, p3, p4, p):
    # define the object 定义对象
    l1 = Getlen(p1, p2)
    l2 = Getlen(p1, p3)
    l3 = Getlen(p2, p3)
    # get the length of two points/获取两点之间直线的长度
    d1 = l1.getlen()
    d2 = l2.getlen()
    d3 = l3.getlen()

def isInTrack(position,trackList):

    x,y=position
    pp = Point(x, y)
    for i in range(len(trackList)):
        p1 = Point(trackList[i][0][0][0],trackList[i][0][0][1])
        p2 = Point(trackList[i][0][1][0],trackList[i][0][1][1])
        p3 = Point(trackList[i][0][2][0],trackList[i][0][2][1])
        p4 = Point(trackList[i][0][3][0],trackList[i][0][3][1])
        if IsPointInMatrix(p1, p2, p3, p4, pp):

            return True


    return False


warnings.filterwarnings("ignore")

plt.ion()


class VGG(nn.Module):
    def __init__(self,num_classes=2):
        super(VGG, self).__init__()
        self.features=nn.Sequential(
            # 第1层卷积 3-->64
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64), # 批归一化操作，为了加速神经网络的收敛过程以及提高训练过程中的稳定性，用于防止梯度消失或梯度爆炸；参数为卷积后输出的通道数；
            nn.ReLU(True),
            # 第2层卷积 64-->64
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 第3层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2,stride=2),
            # 第4层卷积 64-->128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 第5层卷积 128-->128
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 第6层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第7层卷积 128-->256
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第8层卷积 256-->256
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第9层卷积 256-->256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 第10层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第11层卷积 256-->512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第12层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第13层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第14层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第15层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第16层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第17层卷积 512-->512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 第18层池化 图像大小缩小1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier=nn.Sequential(
            # 全连接层512-->4096
            nn.Linear(4608,4096),
            nn.Dropout(),
            # 全连接层4096-->4096
            nn.Linear(4096,4096),
            nn.Dropout(),
            # 全连接层4096-->1000
            nn.Linear(4096,num_classes),
            nn.Dropout()
        )


    def forward(self,a):
        out = self.features(a)# 所得数据为二位数据
        # view功能相当于numpy中resize（）的功能，作用：将一个多行的Tensor,拼接成一行（flatten）
        out = out.view(out.size(0),-1)
        out = self.classifier(out) # 分类数据为一维数据
        return out

class SeqDataset(Dataset):
    def __init__(self, path="record/out200",label=""):


        self.path = path
        self.label=np.load(label)
        # ext = ".npz"
        # files = []
        # for fl in os.listdir(root):
        #     flpath = os.path.join(root, fl)
        #     if (os.path.isfile(flpath) and flpath[-4:] == ext):
        #         files.append(flpath);
        # print(files)
        # files.sort(key=lambda x: int(x[23:-4]))
        # print(files)
        # self.files = files
        # fl=[]
        # self.fl=fl
        # for i in range(len(self.files)):
        #     fl.append(np.load(self.files[i]))
    def __getitem__(self, index):

        img = cv2.imread(self.path+str(index)+".jpg")
        return img.reshape(3,96,96),self.label[index]

    def __len__(self):
        return len(self.label)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7056, 800)
        self.fc2 = nn.Linear(800, 120)
        self.fc3 = nn.Linear(120, 2)

    def forward(self, x):
        input_size = x.size(0)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(input_size,-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class MLP(torch.nn.Module):

    def __init__(self, num_i=5, num_h=100, num_o=10):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x



def valid2(net,test_loader):
    print("begin valid")
    correct = 0
    total = 0
    total_true=0
    total_false=0
    TP=0
    FN=0
    TN=0
    FP=0

    predict_false=0
    for i, data in enumerate(test_loader, 0):
        total=total+len(data[0])
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 优化器清零
        inputs=inputs.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # info2 = info2.to(torch.float32)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        if labels.data==1:
            total_true=total_true+1
            if predicted.data==1:
                TP=TP+1
            else:
                FN=FN+1
        else:
            total_false=total_false+1
            if predicted.data == 0:
                TN=TN+1
            else:
                FP=FP+1
    print("finish valid")
    print(correct/total)
    print(TP)
    print(FN)
    print(TN)
    print(FP)

    return correct/total,TP,FN,TN,FP


def valid(net,test_loader):
    print("begin valid")
    correct = 0
    total = 0

    predict_false=0
    for i, data in enumerate(test_loader, 0):
        total=total+len(data[0])
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()  # 优化器清零
        inputs=inputs.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # info2 = info2.to(torch.float32)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print(correct/total)


    return correct/total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-p', '--path1', default='model/VGG10.pth', help='the path of saved model')
    parser.add_argument('-m', '--model', default='VGG', help='output path, dir\'s name')
    parser.add_argument('-i', '--path2', default="results/VGG10.png", help='The number of episodes should the model plays.')
    args = parser.parse_args()
    saved_model_path=args.path1
    saved_ece_path=args.path2
    device = 'cuda:0'

    acc=[]
    losslist=[]
    prec=[]
    recal=[]
    all=[]
    train_dataset = SeqDataset(path="/home/mao/23Spring/cars/racing_car_data/record/train/img/",label='/home/mao/23Spring/cars/racing_car_data/record/train/labels.npy')
    test_dataset = SeqDataset(path="/home/mao/23Spring/cars/racing_car_data/record/test/img/",label='/home/mao/23Spring/cars/racing_car_data/record/test/labels.npy')
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test_loader2 = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if args.model=="VGG":
        net=VGG()
    else:
        net = Net()
    net=net.to(device)
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)
    for epoch in range(10):
        acc=valid(net, test_loader)
        print(acc)

        running_loss = 0.0
        running_loss2 = 0.0
        correct = 0
        correct2 = 0
        total = 0
        losssum=0
        losssum2=0
        counter=0
        for i, data in enumerate(train_loader, 0):
            total = total + len(data[0])
            inputs, labels = data
            inputs, labels  = Variable(inputs), Variable(labels)

            optimizer.zero_grad()  # 优化器清零
            inputs = inputs.to(torch.float32)
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()  # 优化
            running_loss += loss.item()
            losssum+= loss.item()
            if i % 20 == 19:
                print('[%d %5d] acc: %.3f  loss: %.3f' % (epoch + 1, i + 1, correct / total, running_loss / 20))
                # print('[%d %5d] acc: %.3f' % (epoch + 1, i + 1, ))

                running_loss = 0.0

                correct = 0
                total = 0
            counter=counter+1

        tacc,TP, FN, TN, FP=valid2(net,test_loader2)
        # acc.append(tacc)
        losslist.append(losssum/counter)
        print(acc)
        print(losslist)
        print(TP, FN, TN, FP)
        all.append([TP, FN, TN, FP])
        recal.append(FP/(FP+TN))
        prec.append(TP/(TP+FP))
        print("---")
        print(FP/(FP+TN))
        print(TP/(TP+FP))
        print(all)
        print("-----")
    pred,label_oneh=ece.test(net,test_loader,device)
    ece.draw_reliability_graph(pred,"CNN10.png",label_oneh)
    print('finished training!')
    torch.save(net, "model/CNN10.pth")
