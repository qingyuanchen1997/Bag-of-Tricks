import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import argparse
import scipy
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.optim import lr_scheduler
from myDataset import myDataset
from label_smoothing import LabelSmoothSoftmaxCE
from resnet_bcd import resnet18_bcd

#设置随机数种子
torch.manual_seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)

#训练技巧
parser = argparse.ArgumentParser()
parser.add_argument("--ResNet_BCD", type = bool, default = True, help = "use resnet_BCD or not")
parser.add_argument("--float16", type = bool, default = True, help = "change from float32 to float16 to accelerate training or not")
parser.add_argument("--cosine_decay", type = bool, default = True, help = "use cosine_decay else use step_decay")
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--smoothing_label", type = bool, default = True, help = "use smoothing_label else use one_hot label")
opt = parser.parse_args()
learning_rate = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.ResNet_BCD == True:
    model=resnet18_bcd()
else:
    model = models.resnet18(pretrained=False)
    model.fc=nn.Linear(512, 2)

model = model.to(device)

if opt.float16 == True:
    model=model.half()

transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.Resize([224, 224]),
                                #transforms.RandomHorizontalFlip(p=0.5),
                                #transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

transform_val = transforms.Compose([transforms.ToPILImage(mode=None),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])                                


train_dataset = myDataset('../Cat_Dog/kaggle/my_train/', transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
val_dataset = myDataset('../Cat_Dog/kaggle/my_test/', transform_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False)

loss_smooth_ce = LabelSmoothSoftmaxCE()
loss_ce = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)

if opt.cosine_decay == True:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40,eta_min=1e-6)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)


num_epoch = 150
m = nn.Softmax(dim=1)
training_loss=0
training_cors=0
val_cors=0
num_epoch_no_improvement=0
best_acc=0
patience=15
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
        X, y = data
        X, y = Variable(X.to(device)), Variable(y.to(device))
        if opt.float16 == True:
            X = X.half() 
        pred = model(X)
        output = m(pred.data)
        _, res = torch.max(output, 1)

        optimizer.zero_grad()
        if opt.smoothing_label == True:
            loss = loss_smooth_ce(pred, y)
        else:
            loss = loss_ce(pred, y)
        loss.backward()
        optimizer.step()
        training_loss += loss.data * len(y)
        training_cors += torch.sum(res == y.data)

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
        X=X.half()
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