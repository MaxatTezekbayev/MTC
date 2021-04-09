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
from utils import cae_h_loss, MTC_loss, alter_loss, calculate_B_alter, calculate_singular_vectors_B, knn_distances, sigmoid, Jacobian_for_ALTER, calc_jac, svd_drei
from tqdm import tqdm
import argparse
from collections import Counter
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Implementation of Manifold Tangent Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--dataset', type=str, default="MNIST")

parser.add_argument('--ALTER', type=bool, default=False,
                    help='Train alternating algorithm')

parser.add_argument('--CAEH', type=bool, default=False,
                    help='Train CAE+H')

parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lambd', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.01,
                    help='gamma')

parser.add_argument('--code_size', type=int, default=120,
                    help='dimension of hidden layer')
parser.add_argument('--code_size2', type=int, default=60,
                    help='dimension of hidden layer')

parser.add_argument('--epsilon', type=float, default=0.1,
                    help='std for random noise')

parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')

parser.add_argument('--numlayers', type=int, default=2,
                    help='layers of CAE+H (1 or 2)')

parser.add_argument('--save_dir_for_CAE', type=str, default=None,
                    help='path for saving weights')

parser.add_argument('--pretrained_CAEH', type=str, default=None,
                    help='path to pretrainded state_dict for CAEH. If provided, we will not train CAEH model')

parser.add_argument('--KNN', type=bool, default=False,
                    help='KNN or not')

parser.add_argument('--train_CAEH', type=bool, default=None,
                    help='train_CAEH or not')

# ALTERNATING

parser.add_argument('--M', type=int, default=100,
                    help='the size of the subset for forcing the Jacobian to be of rank not greater than k')
parser.add_argument('--k', type=int, default=40,
                    help='desired rank k for alternating algorithm')
parser.add_argument('--alter_steps', type=int, default=1000,
                    help='steps for alternating algorithm ')
parser.add_argument('--save_dir_for_ALTER', type=str, default=None,
                    help='path for saving weights')


# MTC
parser.add_argument('--MTC', type=bool, default=False,
                    help='train MTC or not')
parser.add_argument('--dM', type=int, default=15,
                    help='number of leading singular vectors')

parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--MTC_epochs', type=float, default=50)
parser.add_argument('--MTC_lr', type=float, default=0.001)

args = parser.parse_args()


batch_size = args.batch_size
k = args.k

if args.CAEH and args.ALTER:
    raise Exception("Select only one: CAEH or ALTER")

if args.dataset == "MNIST":
    image_size = 28
    dimensionality = image_size*image_size
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    # train_dataset = torch.utils.data.Subset(train_dataset, range(0, 700))
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


if args.CAEH:
    if args.numlayers == 2:
        model = CAE2Layer(dimensionality, [args.code_size, args.code_size2])
    else:
        raise Exception("Sorry, numlayers only 2")

elif args.ALTER:
    if args.numlayers == 2:
        model = ALTER2Layer(dimensionality, [args.code_size, args.code_size2])
    else:
        raise Exception("Sorry, numlayers only 2")

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

if args.pretrained_CAEH and args.train_CAEH:
    raise Exception("Select only one: pretrained_CAEH or train_CAEH")

if args.pretrained_CAEH:
    model.load_state_dict(torch.load(args.pretrained_CAEH))

# train CAE+H (ALTER is below)
elif args.train_CAEH is True:
    writer = SummaryWriter('runs/' + "_".join(map(str, ["caeh", args.code_size, args.code_size2, args.learning_rate, args.lambd, args.gamma, args.epsilon])))
    MSELoss = nn.MSELoss()
    for epoch in range(args.epochs):
        train_loss = 0
        test_loss = 0
        MSE_loss = 0
        for step, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, -1).cuda()
            x.requires_grad_(True)
            x_noise = torch.autograd.Variable(x.data + torch.normal(0, args.epsilon, size=[batch_size, dimensionality]).cuda(), requires_grad=True)

            recover, code_data, code_data_noise, Jac = model(x, calculate_jacobian=True)
            loss, loss1 = cae_h_loss(imgs, imgs_noise, recover, code_data,
                                     code_data_noise, args.lambd, args.gamma, batch_size)

            imgs.requires_grad_(False)
            imgs_noise.requires_grad_(False)

            loss.backward()

            train_loss += loss.item()
            MSE_loss += loss1.item()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            for test_imgs, _ in test_loader:
                test_imgs = test_imgs.view(batch_size, -1).cuda()
                test_recover, _ = model(test_imgs)
                test_loss += MSELoss(test_recover, test_imgs).item()

        writer.add_scalar('CAEH/Loss/train', (train_loss / num_batches), epoch)
        writer.add_scalar('CAEH/Loss/train_MSE',
                          (MSE_loss / num_batches), epoch)
        writer.add_scalar('CAEH/Loss/test_MSE',
                          (test_loss / test_num_batches), epoch)

        if epoch % 5 == 0:
            print(epoch, train_loss/num_batches)

    if args.save_dir_for_CAE:
        torch.save(model.state_dict(), args.save_dir_for_CAE)



### ALTER
if args.ALTER:
    writer = SummaryWriter('runs/' + "_".join(map(str, ["alter", args.code_size, args.code_size2, args.learning_rate, args.lambd, args.gamma, args.epsilon])))
    MSELoss = nn.MSELoss()

    B = calculate_B_alter(model, train_z_loader, k, batch_size, first_time = True)
    train_x_iterator = iter(train_loader)
    train_z_iterator = iter(train_z_loader)

    last_step = args.alter_steps-1
    for epoch in range(args.epochs):
        train_loss = 0
        test_loss = 0
        MSE_loss = 0
        B_iter = iter(B)
        for alter_step in tqdm(range(args.alter_steps)):     
            #to always get some batch of x
            try:
                x = next(train_x_iterator)[0]
            except StopIteration:
                train_x_iterator = iter(train_loader)
                x = next(train_x_iterator)[0]

            #to always get some batch of z, b
            try:
                z = next(train_z_iterator)[0]
                b = next(B_iter)
            except StopIteration:
                train_z_iterator = iter(train_z_loader)
                B_iter = iter(B)
                z = next(train_z_iterator)[0]
                b = next(B_iter)

            x = x.view(batch_size, -1).cuda()
            z = z.view(batch_size, -1).cuda()
            b = b.cuda()

            x.requires_grad_(True)
            z.requires_grad_(True)
            x_noise = torch.autograd.Variable(x.data + torch.normal(0, args.epsilon, size=[batch_size, dimensionality]).cuda(), requires_grad=True)

            recover, code_data = model(x)
            _, code_data_noise = model(x_noise)
            _, code_data_z = model(z)
            
            Jac = calc_jac(model, code_data)
            Jac_noise = calc_jac(model, code_data_noise)
            Jac_z = calc_jac(model, code_data_z)

            loss, loss1 = alter_loss(x, recover, Jac, Jac_noise, Jac_z, b, args.lambd, args.gamma)

            x.requires_grad_(False)
            x_noise.requires_grad_(False)
            z.requires_grad_(False)
            if alter_step == last_step:
                loss.backward()
            else:
                loss.backward(retain_graph = True)
            if epoch>0:
                print(alter_step, W1_copy.grad.data.mean())
                model.W1.grad.data += W1_copy.grad.data
                model.W2.grad.data += W2_copy.grad.data
                model.b1.grad.data += b1_copy.grad.data
                model.b2.grad.data += b2_copy.grad.data
                model.b3.grad.data += b3_copy.grad.data

                W1_copy.grad.zero_()
                W2_copy.grad.zero_()
                b1_copy.grad.zero_()
                b2_copy.grad.zero_()
                b3_copy.grad.zero_()
            
            # print(step)
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
        B =[]
        for step, (z, _) in enumerate(tqdm(train_z_loader)):
            z = z.view(batch_size, -1).cuda()
            z.requires_grad_(True)
            # recover_z, code_data_z = model(z, calculate_jacobian = True)

            W1_copy = model.W1.clone().detach().requires_grad_(True).cuda()
            W2_copy = model.W2.clone().detach().requires_grad_(True).cuda()
            b1_copy = model.b1.clone().detach().requires_grad_(True).cuda()
            b2_copy = model.b2.clone().detach().requires_grad_(True).cuda()
            b3_copy = model.b3.clone().detach().requires_grad_(True).cuda()
            b_r_copy = model.b_r.clone().detach().requires_grad_(True).cuda()

            code_data1_z = torch.sigmoid(torch.matmul(z, W1_copy.t()) + b1_copy)
            code_data2_z = torch.sigmoid(torch.matmul(code_data1_z, W2_copy.t()) + b2_copy)
            #decode
            code_data3_z = torch.sigmoid(torch.matmul(code_data2_z, W2_copy) + b3_copy)
            recover_z = torch.sigmoid(torch.matmul(code_data3_z, W1_copy) + b_r_copy)

            # code_data_z = [code_data1, code_data2, code_data3]
            # Jac_z = calc_jac(code_data_z, W1_copy, W2_copy)
            # u, sigma, v = torch.linalg.svd(Jac_z)
            
            #drei
            A_matrix = []
            B_matrix = []
            C_matrix = []
            for i in range(batch_size): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1_z[i], code_data1_z[i]))
                grad_1 = torch.matmul(W1_copy.t(), diag_sigma_prime1)

                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2_z[i], code_data2_z[i]))
                grad_2 = torch.matmul(W2_copy.t(), diag_sigma_prime2)
        
                diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data3_z[i], code_data3_z[i]))
                grad_3 = torch.matmul(W2_copy, diag_sigma_prime3)
        
                A_matrix.append(grad_1)
                B_matrix.append(grad_2)
                C_matrix.append(grad_3)
            A_matrix = torch.reshape(torch.cat(A_matrix, 1),[batch_size, grad_1.shape[0], grad_1.shape[1]])
            B_matrix = torch.reshape(torch.cat(B_matrix, 1),[batch_size, grad_2.shape[0], grad_2.shape[1]])
            C_matrix = torch.reshape(torch.cat(C_matrix, 1),[batch_size, grad_3.shape[0], grad_3.shape[1]])
            U, S, VH = torch.svd(W1_copy)
            for i in range(len(A_matrix)):
                u, s, vh = svd_drei(A_matrix[i], B_matrix[i], C_matrix[i], U, S, VH.T)
                b = torch.matmul(u[:, :k], torch.matmul(torch.diag_embed(s)[:k, :k], vh[:k, :]))
                B.append(b.cpu())
        B= torch.stack(B)
    #end of training

    if args.save_dir_for_ALTER:
        torch.save(model.state_dict(), args.save_dir_for_ALTER)
        torch.save(B, "B_"+args.save_dir_for_ALTER)


# train Manifold Tangent Classifier
if args.MTC is True:
    writer = SummaryWriter('runs/' + "_".join(map(str, ["MTC", args.code_size, args.code_size2, args.learning_rate,
                                                        args.lambd, args.gamma, args.epsilon, args.MTC_lr, args.MTC_epochs, args.beta, args.dM])))
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
        for (imgs, y), u in zip(train_loader, U):
            imgs = imgs.view(batch_size, -1).cuda()
            imgs.requires_grad_(True)
            y = y.cuda()
            u = u.cuda()
            pred = MTC_model(imgs)
            loss, loss1 = MTC_loss(
                pred, y, u, imgs, args.beta, args.batch_size)
            imgs.requires_grad_(False)
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


# if CAEH + KNN
if args.KNN:
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    test_size = 10000
    train_size = 1000
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

    # del model
    # torch.cuda.empty_cache()

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
        with open('results_CAEH.txt', 'a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(MSE_loss/num_batches, args.learning_rate,
                                                                  args.lambd, args.gamma, args.code_size, args.code_size2, args.epsilon, k, accuracies[k]))
