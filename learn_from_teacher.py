import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.optim import lr_scheduler
from teacher_dataset import teacher_dataset
from myDataset import myDataset
import math
import torch.nn.functional as F
from resnet_bcd import resnet18_bcd

torch.manual_seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18_bcd()
model = model.to(device)
batch_size = 128

transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

transform_val = transforms.Compose([transforms.ToPILImage(mode=None),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])                                

T=2 #温度
train_dataset = teacher_dataset('../Cat_Dog/kaggle/my_train/', T, transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = myDataset('../Cat_Dog/kaggle/my_test/', transform_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

loss_kl = torch.nn.KLDivLoss()
loss_ce = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40,eta_min=1e-6)

num_epoch = 150
m = nn.Softmax(dim=1)
training_loss=0
training_cors=0
val_cors=0
num_epoch_no_improvement=0
best_acc=0
patience=25
for epoch in range(0, num_epoch):
    print("lr:",scheduler.get_lr())
    training_loss=0
    training_cors=0
    stop=0
    print("model is in training phase...")
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    for data in train_loader:
        X, y,label_onehot = data
        #整理教师标签形状
        yy=torch.Tensor(2,y[0].size(0))
        yy[0]=y[0]
        yy[1]=y[1]
        y=yy
        yyy=torch.Tensor(y[0].size(0),2)
        for i in range(y[0].size(0)):
            yyy[i,0]=y[0,i]
            yyy[i,1]=y[1,i]
        y=yyy
        y=y/T
        y = F.softmax(y/T,dim=1) 
        X, y, label_onehot = Variable(X.to(device)), Variable(y.to(device)),Variable(label_onehot.to(device))
        
        pred = model(X)
        output = F.log_softmax(pred/T,dim=1)
        _, res = torch.max(output, 1)
        optimizer.zero_grad()
        #按0.5权重计算loss
        loss = 0.5*loss_kl(output, y)*T*T + 0.5*loss_ce(pred,label_onehot)
        loss.backward()
        optimizer.step()
        training_loss += loss.data * len(y)
        training_cors += torch.sum(res == label_onehot.data)

    epoch_loss = training_loss / len(train_dataset)
    epoch_acc =  training_cors*1.0 / len(train_dataset)
    scheduler.step()
    print("Train Loss:{:.4f} Acc:{:.4f}".format(epoch_loss, epoch_acc))
    print("model is in validating phase...")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    val_cors=0
    for data in val_loader:
        X, y = data
        X, y = Variable(X.to(device)), Variable(y.to(device))
        pred = model(X)
        output = m(pred.data)
        _, res = torch.max(output, 1)
        val_cors += torch.sum(res == y.data)

    val_epoch_acc =  val_cors*1.0 / len(val_dataset)
    print("val_Acc:{:.4f}".format(val_epoch_acc))

    if val_epoch_acc > best_acc-0.01:
        num_epoch_no_improvement=0
        print("Validation accuracy increase from {:.4f} to {:.4f}".format(best_acc, val_epoch_acc))
        best_acc=val_epoch_acc
        torch.save({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()

        }, 'checkpoints/'+'baseline'+'.pt')
        print("Saving model!")
    else:
        print("Validation acc does not increase from {:.4f}, num_epoch_no_improvement {}".format(best_acc, num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == patience:
        print("Early Stopping!")
        break