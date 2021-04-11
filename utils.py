import numpy as np
import torch
import torch.nn as nn
import time



def cae_h_loss(imgs, imgs_noise,  recover, code_data, code_data_noise, lambd, gamma, batch_size):
    criterion = nn.MSELoss()
    loss1=criterion(recover, imgs)
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
    
    return loss, loss1

def alter_loss(x, recover, Jac, Jac_noise, Jac_z, b, lambd, gamma):
    criterion = nn.MSELoss()
    loss1 = criterion(recover, x)
    loss2 = torch.mean(torch.sum(torch.pow(Jac - Jac_noise, 2), dim = [1, 2]))
    loss3 = torch.mean(torch.sum(torch.pow(Jac_z - b, 2), dim = [1 ,2]))

    loss = loss1 + (gamma * loss2) + lambd * loss3

    return loss, loss1


def MTC_loss(pred, y, u, imgs, beta, batch_size):
    grad_output=torch.ones(batch_size).cuda()
    
    criterion = nn.CrossEntropyLoss()
    loss1=criterion(pred, y)

    dodx=[]                                                                                        
    for i in range(pred.shape[1]):
        dodx.append(torch.autograd.grad(outputs=pred[:,i], inputs=imgs, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
    dodx=torch.reshape(torch.cat(dodx,1),[batch_size, pred.shape[1], imgs.shape[1]])
    
    omega = torch.mean(torch.sum(torch.pow(torch.matmul(dodx, u),2), dim=[1,2]))
    
    loss=loss1 + beta * omega
    return loss, loss1
 
def svd_product(A, U, S, VH): # A*U*S*VH
    Q, R = torch.qr(torch.matmul(A, U))
    u_temp, s_temp, vh_temp = torch.svd(torch.matmul(R, torch.diag(S)))
    return [torch.matmul(Q, u_temp), s_temp, torch.matmul(vh_temp.T, VH)]

def svd_drei(A, B, C, U, S, VH): # A*B*C*U*S*VH
    U1, S1, VH1 = svd_product(C, U, S, VH)
    U2, S2, VH2 = svd_product(B, U1, S1, VH1)
    return svd_product(A, U2, S2, VH2)

def calculate_B_alter(model, train_z_loader, k, batch_size):
    Bx=[]
    time_model = []
    time_svd = []
    time_b = []
    with torch.no_grad():
        for step, (z, _) in enumerate(train_z_loader):
            z = z.view(batch_size, -1).cuda()
            start_time_model = time.time()
            _, code_data_z, Jac_z = model(z, calculate_jacobian = True)
            time_model.append(time.time() - start_time_model)

            start_time_svd = time.time()
            u, s, v = torch.svd(Jac_z.cpu())
            time_svd.append(time.time() - start_time_svd)

            start_time_b = time.time()
            b = torch.matmul(u[:, :, :k].cuda(), torch.matmul(torch.diag_embed(s)[:, :k, :k].cuda(), torch.transpose(v[:, :, :k],1,2).cuda()))
            time_b.append(time.time() - start_time_b)

            Bx.append(b.cpu())
    print("time_model", np.mean(time_model), "time_svd", np.mean(time_svd), "time_b", np.mean(time_b) )
    return Bx
    
def calculate_singular_vectors_B(model, train_loader, dM, batch_size):
    grad_output=torch.ones(batch_size).cuda()
    U=[]
    for step, (imgs, _) in enumerate(train_loader):
        imgs = imgs.view(batch_size, -1).cuda()
        imgs.requires_grad_(True)
        recover, code_data= model(imgs)
        Jx=[]                                                                                        
        for i in range(code_data.shape[1]):
            Jx.append(torch.autograd.grad(outputs=code_data[:,i], inputs=imgs, grad_outputs=grad_output, retain_graph=True)[0])
        Jx=torch.reshape(torch.cat(Jx,1),[batch_size, code_data.shape[1], imgs.shape[1]])
        u, _, _ = torch.svd(torch.transpose(Jx, 1, 2))
        u=u[:,:,:dM].cpu()
        
        U.append(u)
        if step%100 == 0:
            print("calculating U:", step)
    U = torch.stack(U)
    del Jx
    del u
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