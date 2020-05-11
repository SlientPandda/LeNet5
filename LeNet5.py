# PyTorch：利用PyTorch实现最经典的LeNet卷积神经网络对手写数字进行识别CNN
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Conv1 和 Conv2：卷积层，每个层输出在卷积核（小尺寸的权重张量）和同样尺寸输入区域之间的点积；
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 使用 max 运算执行特定区域的下采样（通常 2x2 像素）；
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))  # 修正线性单元函数，使用逐元素的激活函数 max(0,x)；
        x = F.dropout(x, training=self.training)  # Dropout2D随机将输入张量的所有通道设为零。当特征图具备强相关时，dropout2D 提升特征图之间的独立性；
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)  # 将 Log(Softmax(x)) 函数应用到 n 维输入张量，以使输出在 0 到 1 之间。


# 创建 LeNet 类后，创建对象并移至 GPU
model = LeNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)  # 要训练该模型，我们需要使用带动量的 SGD，学习率为 0.01，momentum 为 0.5。

import os
from torch.autograd import Variable
import torch.nn.functional as F

cuda_gpu = torch.cuda.is_available()


def train(model, epoch, criterion, optimizer, data_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
            model.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset),
                       100. * (batch_idx + 1) / len(data_loader), loss.item()))


from torchvision import datasets, transforms

batch_num_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_num_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_num_size, shuffle=True)


def test(model, epoch, criterion, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
            model.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)  # loss function already averages over batch size
    acc = correct / len(data_loader.dataset)
    #print(acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return (acc, test_loss)


epochs = 40  # 仅仅需要 5 个 epoch（一个 epoch 意味着你使用整个训练数据集来更新训练模型的权重），就可以训练出一个相当准确的 LeNet 模型。
#这段代码检查可以确定文件中是否已有预训练好的模型。有则加载；无则训练一个并保存至磁盘。
if (os.path.isfile('MNIST_net.t7')):
#     print('Loading model')
#     model.load_state_dict(torch.load('MNIST_net.t7', map_location=lambda storage, loc: storage))
#     acc, loss = test(model, 1, criterion, test_loader)
# else:
    print('Training model')  # 打印出该模型的信息。打印函数显示所有层（如 Dropout 被实现为一个单独的层）及其名称和参数。
    for epoch in range(1, epochs + 1):
        train(model, epoch, criterion, optimizer, train_loader)
        acc, loss = test(model, 1, criterion, test_loader)
    torch.save(model.state_dict(), 'MNIST_net.t7')

# print(type(t.cpu().data))  # 以使用 .cpu() 方法将张量移至 CPU（或确保它在那里）。
# # 或当 GPU 可用时（torch.cuda. 可用），使用 .cuda() 方法将张量移至 GPU。你可以看到张量是否在 GPU 上，其类型为 torch.cuda.FloatTensor。
# # 如果张量在 CPU 上，则其类型为 torch.FloatTensor。
# if torch.cuda.is_available():
#     print("Cuda is available")
#     print(type(t.cuda().data))
# else:
#     print("Cuda is NOT available")
#
# if torch.cuda.is_available():
#     try:
#         print(t.data.numpy())
#     except RuntimeError as e:
#         "you can't transform a GPU tensor to a numpy nd array, you have to copy your weight tendor to cpu and then get the numpy array"
# print(type(t.cpu().data.numpy()))
# print(t.cpu().data.numpy().shape)
# print(t.cpu().data.numpy())
#
# data = model.conv1.weight.cpu().data.numpy()
# print(data.shape)
# print(data[:, 0].shape)
#
# kernel_num = data.shape[0]
#
# fig, axes = plt.subplots(ncols=kernel_num, figsize=(2 * kernel_num, 2))
#
# for col in range(kernel_num):
#     axes[col].imshow(data[col, 0, :, :], cmap=plt.cm.gray)
# plt.show()
