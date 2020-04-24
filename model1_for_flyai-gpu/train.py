import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flyai.train_helper import upload_data, download, sava_train_model
from torchvision import datasets, transforms

# 遇到问题不要着急，添加小姐姐微信
"""""""""""""""""""""""""""
"     小姐姐微信flyaixzs   "
"""""""""""""""""""""""""""

# 把本地数据上传到自己的数据盘
# 上传完成之后训练时注释掉即可
upload_data("./data/MNIST.zip", dir_name="/data", overwrite=True)

# 下载数据用于本地训练和线上GPU训练使用
# 已经下载的数据不会重复下载
download("data/MNIST.zip", decompression=True)

MODEL_PATH = "./model"
BATCH_SIZE = 512  # 批次大小
EPOCHS = 20  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
# 开始训练
batch_size = 64
train_dataset = datasets.MNIST(root='data/',
                               train=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='data/',
                              train=False,
                              transform=transforms.ToTensor())

# 装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 24x24
        self.pool = nn.MaxPool2d(2, 2)  # 12x12
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = self.pool(out)  # 12
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), "./model/best.pkl")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

# 训练完成上传自己想要保存的模型到自己的数据盘中
# 或者上传本地自己的模型
sava_train_model(model_file="./model/best.pkl", dir_name="/model", overwrite=False)

# 使用文件加路径名即可下载
# download("model/best.pkl")
