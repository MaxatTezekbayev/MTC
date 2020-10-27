from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Implementation of CAE+H',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lambd', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.01,
                    help='gamma')
parser.add_argument('--Nnoise', type=int, default=30,
                    help='N samples to draw for epsilion')
parser.add_argument('--code_size', type=int, default=60,
                    help='dimension of hidden layer')


parser.add_argument('--epsilon', type=float, default=0.1,
                    help='std for random noise')

parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')


args = parser.parse_args()


image_size = 28
dimensionality = image_size*image_size

dataset1 = datasets.MNIST(
    'data', train=True, download=True, transform=transforms.ToTensor())
dataset2 = datasets.MNIST(
    'data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(
    dataset1, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size)
epoch_size = len(dataset1) // args.batch_size


class CAE(nn.Module):
    def __init__(self, dimensionality, code_size, batch_size, Nnoise):
        super(CAE, self).__init__()
        self.dim = dimensionality
        self.code_size = code_size
        self.batch_size = batch_size
        self.Nnoise = Nnoise
        # parameters
        self.W1 = nn.Parameter(torch.Tensor(dimensionality, code_size))
        self.b1 = nn.Parameter(torch.Tensor(code_size))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, x_noise):
        code_data = self.sigmoid(torch.matmul(
            x.view(-1, self.dim), self.W1) + self.b1)
        recover = self.sigmoid(torch.matmul(code_data, self.W1.t()) + self.b_r)
        recover = recover.view(*x.shape)
        code_data_noise = torch.sigmoid(torch.matmul(
            x_noise.view(-1, self.Nnoise, self.dim), self.W1) + self.b1)

        return recover, code_data, code_data_noise


def cae_h_loss(imgs, imgs_noise,  recover, code_data, code_data_noise, lambd, gamma):
    criterion = nn.MSELoss()
    loss1 = criterion(recover, imgs)

    Jx = torch.autograd.grad(outputs=code_data, inputs=imgs, grad_outputs=torch.ones_like(
        code_data), create_graph=True)[0]
    Jx_noise = torch.autograd.grad(outputs=code_data_noise, inputs=imgs_noise,
                                   grad_outputs=torch.ones_like(code_data_noise), create_graph=True)[0]

    loss2 = torch.mean(torch.sum(torch.pow(Jx, 2), dim=[1, 2, 3]))
    loss3 = torch.mean(torch.mean(
        torch.sum(torch.pow(Jx - Jx_noise, 2), dim=[2, 3]), dim=1))

    loss = loss1 + (lambd*loss2) + gamma*loss3

    return loss


model = CAE(dimensionality, args.code_size, args.batch_size, args.Nnoise)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


for i in range(args.epoch):
    train_loss = 0
    for step, (imgs, _) in enumerate(train_loader):
        imgs = imgs.cuda()
        imgs.requires_grad_(True)
        imgs_noise = torch.autograd.Variable(imgs.data + torch.normal(0, args.epsilon, size=[
                                             args.batch_size, args.Nnoise, image_size, image_size]).cuda(), requires_grad=True)

        recover, code_data, code_data_noise = model(imgs, imgs_noise)
        loss = cae_h_loss(imgs, imgs_noise, recover, code_data,
                          code_data_noise, args.lambd, args.gamma)

        imgs.requires_grad_(False)
        imgs_noise.requires_grad_(False)

        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        optimizer.zero_grad()
    if i % 10 == 0:
        print(i, train_loss/epoch_size)


# KNN

def euclidean_distance(img_a, img_b):
    '''Finds the distance between 2 images: img_a, img_b'''
    # element-wise computations are automatically handled by numpy
    return np.sum((img_a - img_b) ** 2)


def find_majority(labels):
    '''Finds the majority class/label out of the given labels'''
    # defaultdict(type) is to automatically add new keys without throwing error.
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


dataset1 = datasets.MNIST(
    'data', train=True, download=True, transform=transforms.ToTensor())
dataset2 = datasets.MNIST(
    'data', train=False, download=True, transform=transforms.ToTensor())

dataset1 = torch.utils.data.Subset(dataset1, range(0, 1000))
dataset2 = torch.utils.data.Subset(dataset2, range(0, 10000))
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1))
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2))

train_images = next(iter(train_loader))[0].numpy()
train_labels = next(iter(train_loader))[1].numpy()
test_images = next(iter(test_loader))[0].numpy()
test_labels = next(iter(test_loader))[1].numpy()


def sigmoid(x):
    return 1. / (1+np.exp(-x))


def new_euclidean_distance(img_a, img_b):
    img_a = np.reshape(img_a, (1, -1))
    img_b = np.reshape(img_b, (1, -1))
    img_a = sigmoid(np.matmul(img_a, cur_W) + cur_b)
    img_b = sigmoid(np.matmul(img_b, cur_W) + cur_b)
    return np.sum((img_a - img_b) ** 2)


def predict(k, train_images, train_labels, test_images):
    '''
    Predicts the new data-point's category/label by 
    looking at all other training labels
    '''
    # distances contains tuples of (distance, label)
    distances = [(euclidean_distance(test_image, image), label)
                 for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distances

    def compare(distance): return distance[0]
    by_distances = sorted(distances, key=compare)
    # extract only k closest labels
    k_labels = [label for (_, label) in by_distances[:k]]
    # return the majority voted label
    return find_majority(k_labels)


def new_predict(k, train_images, train_labels, test_images):
    '''
    Predicts the new data-point's category/label by 
    looking at all other training labels
    '''
    # distances contains tuples of (distance, label)
    distances = [(new_euclidean_distance(test_image, image), label)
                 for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distances

    compare = lambda distance: distance[0]
    by_distances = sorted(distances, key=compare)
    # extract only k closest labels
    k_labels = [label for (_, label) in by_distances[:k]]
    # return the majority voted label
    return find_majority(k_labels)


cur_W = model.W1.cpu().detach().numpy()
cur_b = model.b1.cpu().detach().numpy()


del model
# Predicting and printing the accuracy
for k in range(5, 10, 4):
    i = 0
    total_correct = 0
    for test_image in test_images:
        pred = new_predict(k, train_images, train_labels, test_image)
        if pred == test_labels[i]:
            total_correct += 1
        acc = (total_correct / (i+1)) * 100
        if i % 100 == 0:
            print('test image['+str(i)+']', '\tpred:', pred, '\torig:',
                  test_labels[i], '\tacc:', str(round(acc, 2))+'%')
        i += 1

    with open('results.txt','a') as f:
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(args.learning_rate, args.lambd, args.gamma, args.code_size, args.epsilon, k, acc ))


