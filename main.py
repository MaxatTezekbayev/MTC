from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from models import CAE2Layer, MTC, ALTER2Layer
from utils import cae_h_loss, MTC_loss, alter_loss, calculate_B_alter, calculate_singular_vectors_B, knn_distances, sigmoid
from tqdm import tqdm
import argparse
from collections import Counter
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Implementation of Manifold Tangent Classifier and Alternating Scheme',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100,
                    help='max epoch')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--lambd', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.01, help='gamma')
parser.add_argument('--numlayers', type=int, default=2, help='layers of CAE+H (1 or 2)')
parser.add_argument('--code_size', type=int, default=120, help='dimension of 1st hidden layer')
parser.add_argument('--code_size2', type=int, default=60, help='dimension of 2nd hidden layer')

parser.add_argument('--save_autoencoder_path', type=str, default=None,
                    help='path for saving weights')

parser.add_argument('--pretrained_autoencoder_path', type=str, default=None,
                    help='path to pretrainded state_dict for autoencoder. If provided, we will not train autoencoder model')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='std for random noise')
parser.add_argument('--CAEH', type=bool, default=False, help='choose CAE+H autoencoder')
parser.add_argument('--ALTER', type=bool, default=False, help='choose alternating algorithm autoencoder')


# ALTERNATING specific arguments

parser.add_argument('--M', type=int, default=100,
                    help='the size of the subset for forcing the Jacobian to be of rank not greater than k')
parser.add_argument('--k', type=int, default=40,
                    help='desired rank k for alternating algorithm')
parser.add_argument('--alter_steps', type=int, default=1000,
                    help='steps for alternating algorithm ')
parser.add_argument('--save_dir_for_ALTER', type=str, default=None,
                    help='path for saving weights')
parser.add_argument('--optimized_SVD', type=bool, default=None,
                    help='use optimized SVD or not')


# MTC specific arguments
parser.add_argument('--MTC', type=bool, default=False,
                    help='train MTC or not')
parser.add_argument('--MTC_save_path', type=str, default=None,
                    help='path to save MTC weights')
parser.add_argument('--dM', type=int, default=15,
                    help='number of leading singular vectors')

parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--MTC_epochs', type=int, default=50)
parser.add_argument('--MTC_lr', type=float, default=0.001)

# KNN specific arguments
parser.add_argument('--KNN', type=bool, default=False,
                    help='run KNN or not')
parser.add_argument('--KNN_train_size', type=int, default=10000,
                    help='number of points in train set')
parser.add_argument('--KNN_test_size', type=int, default=1000,
                    help='number of points in test set')
args = parser.parse_args()

batch_size = args.batch_size
k = args.k

assert args.CAEH ^ args.ALTER, "Select only one: CAEH or ALTER" #xor
assert args.numlayers==1 or args.numlayers==2, "Sorry, number of layers 1 or 2"

if args.dataset == "MNIST":
    image_size = 28
    dimensionality = image_size*image_size
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if args.ALTER:
        # add z
        indices = torch.randperm(len(train_dataset))[:args.M]
        train_z_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices))
else:
    raise Exception("Sorry, only MNIST")


num_batches = len(train_dataset) // batch_size
test_num_batches = len(test_dataset) // batch_size



if args.CAEH is True:
    if args.numlayers == 2:
        model = CAE2Layer(dimensionality, [args.code_size, args.code_size2])
    elif args.numlayers == 1:
        pass
elif args.ALTER is True:
    if args.numlayers == 2:
        model = ALTER2Layer(dimensionality, [args.code_size, args.code_size2])
    elif args.numlayers == 1:
        pass

#if  pretrained_autoencoder_path load pretrained weights of autoencoder  
if args.pretrained_autoencoder_path:
    model.load_state_dict(torch.load(args.pretrained_autoencoder_path))

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model_name=None
if args.pretrained_autoencoder_path is None:
    model_name="caeh" if args.CAEH else 'ALTER'
    writer = SummaryWriter('runs/' + "_".join(map(str, [model_name, args.code_size, args.code_size2, args.lr, args.lambd, args.gamma, args.epsilon])))
    MSELoss = nn.MSELoss()
    # train CAE+H (ALTER is below)
    if args.CAEH is True:
        for epoch in range(args.epochs):
            train_loss = 0
            test_loss = 0
            MSE_loss = 0
            for step, (x, _) in enumerate(train_loader):
                x = x.view(batch_size, -1).cuda()
                x.requires_grad_(True)
                x_noise = torch.autograd.Variable(x.data + torch.normal(0, args.epsilon, size=[batch_size, dimensionality]).cuda(), requires_grad=True)

                recover, code_data, Jac = model(x, calculate_jacobian=True)
                _, code_data_noise, Jac_noise = model(x_noise, calculate_jacobian=True)
                loss, loss1 = cae_h_loss(x, recover, Jac, Jac_noise, args.lambd, args.gamma)

                x.requires_grad_(False)
                x_noise.requires_grad_(False)

                loss.backward()

                train_loss += loss.item()
                MSE_loss += loss1.item()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                for test_x, _ in test_loader:
                    test_x = test_x.view(batch_size, -1).cuda()
                    test_recover, _ = model(test_x)
                    test_loss += MSELoss(test_recover, test_x).item()

            writer.add_scalar('CAEH/Loss/train', (train_loss / num_batches), epoch)
            writer.add_scalar('CAEH/Loss/train_MSE', (MSE_loss / num_batches), epoch)
            writer.add_scalar('CAEH/Loss/test_MSE', (test_loss / test_num_batches), epoch)

            print("CAEH", epoch, train_loss/num_batches, test_loss/test_num_batches)

        if args.save_autoencoder_path:
            torch.save(model.state_dict(), args.save_autoencoder_path)



    ### ALTER
    if args.ALTER is True:
        assert args.M % args.batch_size == 0, "batch_size should be a divisor of both train size and args.M"
        #initialize B with 0-s
        B = torch.zeros((len(train_z_loader),1))
        train_x_iterator = iter(train_loader)
        z_b_iter = iter(zip(train_z_loader,B))
        for epoch in range(args.epochs):
            train_loss = 0
            test_loss = 0
            MSE_loss = 0
            for alter_step in tqdm(range(args.alter_steps)):     
                #to always get some batch of x
                try:
                    x = next(train_x_iterator)[0]
                except StopIteration:
                    train_x_iterator = iter(train_loader)
                    x = next(train_x_iterator)[0]

                #to always get some batch of z, b
                try:
                    (z, _), b = next(z_b_iter)
                except StopIteration:
                    z_b_iter = iter(zip(train_z_loader,B))
                    (z, _), b = next(z_b_iter)

                x = x.view(batch_size, -1).cuda()
                z = z.view(batch_size, -1).cuda()
                b = b.cuda()

                x.requires_grad_(True)
                z.requires_grad_(True)
                x_noise = torch.autograd.Variable(x.data + torch.normal(0, args.epsilon, size=[batch_size, dimensionality]).cuda(), requires_grad=True)

                recover, code_data, Jac = model(x, calculate_jacobian = True)
                _, code_data_noise, Jac_noise = model(x_noise, calculate_jacobian = True)
                _, code_data_z, Jac_z = model(z, calculate_jacobian = True)

                loss, loss1 = alter_loss(x, recover, Jac, Jac_noise, Jac_z, b, args.lambd, args.gamma)

                x.requires_grad_(False)
                x_noise.requires_grad_(False)
                z.requires_grad_(False)
                
                loss.backward()

                train_loss += loss.item()
                MSE_loss += loss1.item()

                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                for test_x, _ in test_loader:
                    test_x = test_x.view(batch_size, -1).cuda()
                    test_recover, _ = model(test_x)
                    test_loss += MSELoss(test_recover, test_x).item()

            writer.add_scalar('ALTER/Loss/train', (train_loss / num_batches), epoch)
            writer.add_scalar('ALTER/Loss/train_MSE', (MSE_loss / num_batches), epoch)
            writer.add_scalar('ALTER/Loss/test_MSE', (test_loss / test_num_batches), epoch)
            print(epoch, train_loss/num_batches)
            #calculate B
            B =calculate_B_alter(model, train_z_loader, k, batch_size, args.optimized_SVD)
        #end of training

        if args.save_autoencoder_path:
            torch.save(model.state_dict(), args.save_autoencoder_path)
            torch.save(B, "B_"+args.save_autoencoder_path)


# train Manifold Tangent Classifier
if args.MTC is True:
    autoencoder_model_name = args.pretrained_autoencoder_path if args.pretrained_autoencoder_path else args.save_autoencoder_path
    #if both pretrained_autoencoder_path and save_autoencoder_path are None, than:
    if autoencoder_model_name is None:
        autoencoder_model_name = "_".join(map(str, [model_name, args.code_size, args.code_size2, args.lr, args.lambd, args.gamma, args.epsilon]))
    writer = SummaryWriter('runs/' + "_".join(map(str, ["MTC", autoencoder_model_name, args.MTC_lr, args.MTC_epochs, args.beta, args.dM])))
    if args.ALTER:
        U = torch.load("B_"+args.pretrained_autoencoder_path)
    else:
        U = calculate_singular_vectors_B(model, train_loader, args.dM, batch_size)

    number_of_classes = len(train_dataset.classes)
    MTC_model = MTC(model, number_of_classes)
    MTC_model.cuda()
    optimizer = optim.Adam(MTC_model.parameters(), lr=args.MTC_lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.MTC_epochs):
        train_loss = 0
        CE_loss = 0
        correct = 0
        test_loss = 0
        test_correct = 0
        for (x, y), u in zip(train_loader, U):
            x = x.view(batch_size, -1).cuda()
            x.requires_grad_(True)
            y = y.cuda()
            u = u.cuda()
            pred = MTC_model(x)
            loss, loss1 = MTC_loss(pred, y, u, x, args.beta, args.batch_size)
            x.requires_grad_(False)
            loss.backward()
            train_loss += loss.item()
            CE_loss += loss1.item()
            _, preds = torch.max(pred, 1)
            correct += torch.sum(preds == y.data).item()

            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            for test_input, test_labels in test_loader:
                test_input = test_input.view(batch_size, -1).cuda()
                test_labels = test_labels.cuda()
                test_outputs = MTC_model(test_input)
                test_loss += criterion(test_outputs, test_labels).item()
                _, test_preds = torch.max(test_outputs, 1)
                test_correct += torch.sum(test_preds ==
                                          test_labels.data).item()

        writer.add_scalar('MTC/Loss/train', (train_loss / num_batches), epoch)
        writer.add_scalar('MTC/Loss/train_CE', (CE_loss / num_batches), epoch)
        writer.add_scalar('MTC/Loss/test_CE', (test_loss / test_num_batches), epoch)
        writer.add_scalar('MTC/Acc/train', (correct / (num_batches*batch_size)), epoch)
        writer.add_scalar('MTC/Acc/test', (test_correct / (test_num_batches*batch_size)), epoch)
        print(epoch, train_loss/num_batches, CE_loss/num_batches, (test_loss / test_num_batches), correct / (num_batches*batch_size), test_correct / (test_num_batches*batch_size))
    
    if args.MTC_save_path is not None:
        torch.save(model.state_dict(), args.MTC_save_path)


# if CAEH + KNN
if args.KNN:
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    test_size = args.KNN_train_size
    train_size = args.KNN_test_size
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, train_size))
    test_dataset = torch.utils.data.Subset(test_dataset, range(0, test_size))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

    train_images = next(iter(train_loader))[0].numpy()
    train_labels = next(iter(train_loader))[1].numpy()
    test_images = next(iter(test_loader))[0].numpy()
    test_labels = next(iter(test_loader))[1].numpy()

    train_images = np.reshape(train_images, (train_size, -1))
    test_images = np.reshape(test_images, (test_size, -1))

    weights = None
    if args.numlayers == 1:
        cur_W1 = model.W1.cpu().detach().numpy()
        cur_b1 = model.b1.cpu().detach().numpy()
        weights = [[cur_W1, cur_b1]]
    elif args.numlayers == 2:
        cur_W1 = model.W1.cpu().detach().numpy()
        cur_b1 = model.b1.cpu().detach().numpy()
        cur_W2 = model.W2.cpu().detach().numpy()
        cur_b2 = model.b2.cpu().detach().numpy()
        weights = [[cur_W1, cur_b1], [cur_W2, cur_b2]]

    # encode images
    for W, b in weights:
        train_images = sigmoid(np.matmul(train_images, W.T) + b)
        test_images = sigmoid(np.matmul(test_images, W.T) + b)

    # Predicting and printing the accuracy

    ks = np.arange(1, 20, 2)

    i = 0
    total_correct = {}
    for k in ks:
        total_correct[k] = 0

    for test_image in test_images:
        top_n_labels = knn_distances(
            train_images, train_labels, test_image, n_top=20)
        for k in ks:
            pred = Counter(top_n_labels[:k]).most_common(1)[0][0]
            if pred == test_labels[i]:
                total_correct[k] += 1
        if i % 4000 == 0:
            print('test image['+str(i)+']')
        i += 1

    accuracies = {k: round((v/i) * 100, 2) for k, v in total_correct.items()}

    for k in ks:
        writer.add_scalar('K_acc', accuracies[k], k)
        with open('results_CAEH_tied_0.txt', 'a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(MSE_loss/num_batches, args.lr,
                                                                  args.lambd, args.gamma, args.code_size, args.code_size2, args.epsilon, k, accuracies[k]))
