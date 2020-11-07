from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from models import *
from utils import knn_distances, sigmoid
import argparse
from collections import Counter

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Implementation of CAE+H',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lambd', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.01,
                    help='gamma')

parser.add_argument('--code_size', type=int, default=60,
                    help='dimension of hidden layer')
parser.add_argument('--code_size2', type=int, default=None,
                    help='dimension of hidden layer')

parser.add_argument('--epsilon', type=float, default=0.1,
                    help='std for random noise')

parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

parser.add_argument('--numlayers', type=int, default=1,
                    help='layers of CAE+H (1 or 2)')

parser.add_argument('--save_dir_for_CAE', type=str, default=None,
                    help='directory for saving weights')


parser.add_argument('--KNN', type=bool, default=False,
                    help='KNN or not')

parser.add_argument('--MCT', type=bool, default=False,
                    help='train MCT or not')


args = parser.parse_args()


image_size = 28
dimensionality = image_size*image_size
epoch_size = len(dataset1) // args.batch_size



dataset1 = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
dataset2 = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.batch_size)


def cae_h_loss(imgs, imgs_noise,  recover, code_data, code_data_noise, lambd, gamma):
    criterion = nn.MSELoss()
    loss1=criterion(recover, imgs)
    #incorrect:
#     Jx = torch.autograd.grad(outputs=code_data, inputs=imgs, grad_outputs=torch.ones_like(
#         code_data), create_graph=True)[0]
#     Jx_noise = torch.autograd.grad(outputs=code_data_noise, inputs=imgs_noise,
#                                    grad_outputs=torch.ones_like(code_data_noise), create_graph=True)[0]
    
    # old variant (slow):
#     Jx=[]
#     for i in range(batch_size):
#         for j in range(code_data.shape[1]):
#             Jx.append(torch.autograd.grad(outputs=code_data[i][j], inputs=imgs, retain_graph=True, create_graph=True)[0][i])
#     Jx=torch.reshape(torch.cat(Jx),[batch_size, code_data.shape[1], imgs.shape[1]])
    
#     Jx_noise=[]
#     for i in range(batch_size):
#         for j in range(code_data.shape[1]):
#             Jx_noise.append(torch.autograd.grad(outputs=code_data_noise[i][j], inputs=imgs_noise, retain_graph=True, create_graph=True)[0][i])
#     Jx_noise=torch.reshape(torch.cat(Jx_noise),[batch_size, code_data_noise.shape[1], imgs_noise.shape[1]])
    
    
    #new variant (faster)
    grad_output=torch.ones(batch_size).cuda()
    Jx=[]                                                                                        
    for i in range(code_data.shape[1]):
        Jx.append(torch.autograd.grad(outputs=code_data[:,i], inputs=imgs, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
    Jx=torch.reshape(torch.cat(Jx,1),[batch_size, code_data.shape[1], imgs.shape[1]])
    
    Jx_noise=[]                                                                                        
    for i in range(code_data_noise.shape[1]):
        Jx_noise.append(torch.autograd.grad(outputs=code_data_noise[:,i], inputs=imgs_noise, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
    Jx_noise=torch.reshape(torch.cat(Jx_noise,1),[batch_size, code_data_noise.shape[1], imgs_noise.shape[1]])

    loss2 = torch.mean(torch.sum(torch.pow(Jx,2), dim=[1,2]))
    loss3 = torch.mean(torch.sum(torch.pow(Jx - Jx_noise,2),dim=[1,2]))
    loss = loss1 + (lambd*loss2) + gamma*loss3
    
    return loss

if args.numlayers==1:
    model = CAE1Layer(dimensionality, args.code_size)
elif args.numlayers==2:
    model = CAE2Layer(dimensionality, [args.code_size, args.code_size2])
else:
    raise Exception("Sorry, numlayers only 1 or 2")

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


for i in range(args.epochs):
    train_loss = 0
    for step, (imgs, _) in enumerate(train_loader):
        imgs = imgs.view(batch_size, -1).cuda()
        imgs.requires_grad_(True)
        imgs_noise = torch.autograd.Variable(imgs.data + torch.normal(0, epsilon, size=[batch_size, dimensionality]).cuda(),requires_grad=True)

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

if args.save_dir_for_CAE:
    torch.save(model.state_dict(), args.save_dir_for_CAE)

if args.MCT:
    pass


if args.KNN:
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    dataset2 = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    test_size=10000
    train_size=1000
    dataset1=torch.utils.data.Subset(dataset1,range(0,train_size))
    dataset2=torch.utils.data.Subset(dataset2,range(0,test_size))
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1))
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2))

    train_images = next(iter(train_loader))[0].numpy()
    train_labels = next(iter(train_loader))[1].numpy()
    test_images = next(iter(test_loader))[0].numpy()
    test_labels = next(iter(test_loader))[1].numpy()

    train_images=np.reshape(train_images, (train_size, -1))
    test_images=np.reshape(test_images, (test_size, -1))

    weights=None
    if args.numlayers==1:
        cur_W1 = model.W1.cpu().detach().numpy()
        cur_b1 = model.b1.cpu().detach().numpy()
        weights=[[cur_W1, cur_b1]]
    elif args.numlayers==2:
        cur_W1 = model.W1.cpu().detach().numpy()
        cur_b1 = model.b1.cpu().detach().numpy()
        cur_W2 = model.W2.cpu().detach().numpy()
        cur_b2 = model.b2.cpu().detach().numpy()
        weights=[[cur_W1, cur_b1],[cur_W2, cur_b2]]

    #encode images
    for W,b in weights:
        train_images = sigmoid(np.matmul(train_images, W.T) + b)
        test_images = sigmoid(np.matmul(train_images, W.T) + b)

    # del model
    # torch.cuda.empty_cache()

    # Predicting and printing the accuracy
    
    ks=np.arange(1,20,2)

    i = 0
    total_correct={}
    for k in ks:
        total_correct[k]=0
        
    for test_image in test_images:
        top_n_labels = knn_distances(train_images, train_labels, test_image, n_top=20)
        for k in ks:
            pred = Counter(top_n_labels[:k]).most_common(1)[0][0]
            if pred == test_labels[i]:
                total_correct[k] += 1
        if i%4000 == 0:
            print('test image['+str(i)+']')
        i += 1

    accuracies = {k: round((v/i) * 100, 2) for k,v in total_correct.items()}

    for k in ks:
        with open('results_CAEH.txt','a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(args.learning_rate, args.lambd, args.gamma, args.code_size, args.code_size2, args.epsilon, k, accuracies[k]))


