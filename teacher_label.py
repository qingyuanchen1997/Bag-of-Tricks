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
from myDataset import myDataset
from label_smoothing import LabelSmoothSoftmaxCE
from resnet_bcd import resnet50_bcd

torch.manual_seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50_bcd()
model = model.to(device)
model = model.half()

#加载教师模型
checkpoint = torch.load('checkpoints/teacher.pt')
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.","")] = state_dict[key]
model.load_state_dict(state_dict)

model = model.to(device)
batch_size = 128

transform_val = transforms.Compose([transforms.ToPILImage(mode=None),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])                                

train_dataset = myDataset('../Cat_Dog/kaggle/my_train/', transform_val)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

m = nn.Softmax(dim=1)
train_cors=0
num_epoch_no_improvement=0
best_acc=0
train_epoch_acc=0

preds = torch.zeros(len(train_dataset), 2)
ind=0
model.eval()
for param in model.parameters():
    param.requires_grad = False
train_cors=0
for data in train_loader:
    X, y = data
    X, y = Variable(X.to(device)), Variable(y.to(device))
    X=X.half()
    pred = model(X)
    preds[ind:ind + len(y), :] = pred[0:len(y),:]
    ind=ind + len(y)
    output = m(pred.data)
    _, res = torch.max(output, 1)
    train_cors += torch.sum(res == y.data)

train_epoch_acc =  train_cors*1.0 / len(train_dataset)
print("train_Acc:{:.4f}".format(train_epoch_acc))

#保存教师标签
f = open('preds.txt', mode='w')
preds_numpy = preds.numpy()
for i in range(len(preds)):
    content = str(preds_numpy[i,0])+','+str(preds_numpy[i,1]) + '\n'
    f.write(content)
