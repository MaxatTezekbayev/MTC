import numpy as np
import torch
import torch.nn as nn
import time



def cae_h_loss(x, recover, Jac, Jac_noise, lambd, gamma):
    criterion = nn.MSELoss()
    loss1=criterion(recover, x)

    loss2 = torch.mean(torch.sum(torch.pow(Jac,2), dim=[1,2]))
    loss3 = torch.mean(torch.sum(torch.pow(Jac - Jac_noise,2),dim=[1,2]))
    loss = loss1 + (lambd*loss2) + gamma*loss3
    return loss, loss1

def alter_loss(x, recover, Jac, Jac_noise, Jac_z, b, lambd, gamma):
    criterion = nn.MSELoss()
    loss1 = criterion(recover, x)
    loss2 = torch.mean(torch.sum(torch.pow(Jac - Jac_noise, 2), dim = [1, 2]))
    loss3 = torch.mean(torch.sum(torch.pow(Jac_z - b, 2), dim = [1 ,2]))
    loss = loss1 + (gamma * loss2) + lambd * loss3
    return loss, loss1


def MTC_loss(pred, y, u, x, beta, batch_size):
    grad_output=torch.ones(batch_size).cuda()
    criterion = nn.CrossEntropyLoss()
    loss1=criterion(pred, y)

    dodx=[]                                                                                        
    for i in range(pred.shape[1]):
        dodx.append(torch.autograd.grad(outputs=pred[:,i], inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
    dodx=torch.reshape(torch.cat(dodx,1),[batch_size, pred.shape[1], x.shape[1]])
    
    omega = torch.mean(torch.sum(torch.pow(torch.matmul(dodx, u),2), dim=[1,2]))
    
    loss=loss1 + beta * omega
    return loss, loss1
 
def svd_product(A, U, S, VH): # A*U*S*VH
    Q, R = torch.qr(torch.matmul(A.cuda(), U.cuda()).cpu())
    u_temp, s_temp, vh_temp = torch.svd(torch.matmul(R.cuda(), torch.diag(S.cuda())).cpu())
    return [torch.matmul(Q.cuda(), u_temp.cuda()), s_temp.cuda(), torch.matmul(vh_temp.T.cuda(),VH.cuda())]

def svd_drei(A, B, C, D): # A*B*C*D
    U_temp, S_temp, VH_temp = torch.svd(torch.matmul(C, D).cpu())
    return svd_product(torch.matmul(A, B), U_temp.cuda(), S_temp.cuda(), VH_temp.T.cuda())


def calculate_B_alter(model, train_z_loader, k, batch_size, optimized_SVD):
    Bx=[]  
    with torch.no_grad():
        for step, (z, _) in enumerate(train_z_loader):
            z = z.view(batch_size, -1).cuda()

            if optimized_SVD:
                Bx_batch = []
                _, code_data_z, A_matrix, B_matrix, C_matrix = model(z, calculate_jacobian = False, calculate_DREI = True)
                U=[]
                S=[]
                VH=[]
                W4 = model.W1.t().clone()
                for i in range(len(A_matrix)):
                    u, s, vh = svd_drei(W4, C_matrix[i], B_matrix[i],  A_matrix[i])
                    U.append(u)
                    S.append(s)
                    VH.append(vh)
                U = torch.stack(U)
                S = torch.stack(S)
                VH = torch.stack(VH)
                Bx_batch = torch.matmul(U[:, :, :k], torch.matmul(torch.diag_embed(S)[:, :k, :k], VH[:, :k, :]))
            else:
                _, code_data_z, Jac_z = model(z, calculate_jacobian = True)
                U, S, V = torch.svd(Jac_z.cpu())
                Bx_batch = torch.matmul(U[:, :, :k], torch.matmul(torch.diag_embed(S)[:, :k, :k], torch.transpose(V[:, :, :k],1,2)))

            Bx.append(Bx_batch)
    return Bx
    
def calculate_singular_vectors_B(model, train_loader, dM, batch_size):
    U=[]
    with torch.no_grad():
        for step, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, -1).cuda()
            x.requires_grad_(True)
            recover, code_data, Jac = model(x, calculate_jacobian = True)
            u, _, _ = torch.svd(torch.transpose(Jac.cpu(), 1, 2))
            U.append(u[:,:,:dM].cpu())
            if step%100 == 0:
                print("calculating U:", step)
        U = torch.stack(U)
    return U



def sigmoid(x):
    return 1. / (1+np.exp(-x))


def euclidean_distance(img_a, img_b):
    '''Finds the distance between 2 images: img_a, img_b'''
    # element-wise computations are automatically handled by numpy
    return np.sum((img_a - img_b) ** 2)


def knn_distances(train_images, train_labels, test_image, n_top):
    '''
    returns n_top distances and labels for given test_image
    '''
    # distances contains tuples of (distance, label)
    distances = [(euclidean_distance(test_image, image), label)
                 for (image, label) in zip(train_images, train_labels)]
    # sort the distances list by distances

    compare = lambda distance: distance[0]
    by_distances = sorted(distances, key=compare)
    top_n_labels = [label for (_, label) in by_distances[:n_top]]
    return top_n_labels