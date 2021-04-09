import numpy as np
import torch
import torch.nn as nn



def Jacobian_for_ALTER(model, code_data):
    Jac=[]
    for i in range(code_data[0].shape[0]): #batch_size
        diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data[0][i], code_data[0][i]))
        grad_1 = torch.matmul(model.W1.T, diag_sigma_prime1)

        diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data[1][i], code_data[1][i]))
        grad_2 = torch.matmul(model.W2.T, diag_sigma_prime2)

        diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data[2][i], code_data[2][i]))
        grad_3 = torch.matmul(model.W2, diag_sigma_prime3)

        grad_4 = model.W1
        Jac.append(torch.matmul(grad_1, torch.matmul(grad_2, torch.matmul(grad_3, grad_4))))
    Jac = torch.reshape(torch.cat(Jac,1),[code_data[0].shape[0], model.W1.shape[1],  model.W1.shape[1]]) #[batch_size, recover.shape[1], x.shape[1]]
    return Jac

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
    # loss1 = criterion(recover, x)
    # loss2 = torch.mean(torch.sum(torch.pow(Jac - Jac_noise, 2), dim = [1, 2]))
    # loss3 = torch.mean(torch.sum(torch.pow(Jac_z - b, 2), dim = [1 ,2]))

    # loss = loss1 + (gamma * loss2) + lambd * loss3
    loss1 = torch.mean(torch.sum(torch.pow(b, 2)))
    loss = loss1
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

def calc_jac(model, code_data):
    batch_size = code_data[0].shape[0]
    Jac = []
    for i in range(batch_size): 
        diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data[0][i], code_data[0][i]))
        grad_1 = torch.matmul(model.W1.t(), diag_sigma_prime1)

        diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data[1][i], code_data[1][i]))
        grad_2 = torch.matmul(model.W2.t(), diag_sigma_prime2)

        diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data[2][i], code_data[2][i]))
        grad_3 = torch.matmul(model.W2, diag_sigma_prime3)

        grad_4 = model.W1
        Jac.append(torch.matmul(grad_1, torch.matmul(grad_2, torch.matmul(grad_3, grad_4))))
    Jac = torch.reshape(torch.cat(Jac,1),[batch_size, model.W1.shape[1], model.W1.shape[1]])
    return Jac

def calculate_B_alter(model, train_z_loader, k, batch_size):
    Bx=[]
    for step, (z, _) in enumerate(train_z_loader):
        z = z.view(batch_size, -1).cuda()
        z.requires_grad_(True)

        W1_copy = model.W1.clone()
        W2_copy = model.W2.clone()
        b1_copy = model.b1.clone()
        b2_copy = model.b2.clone()
        b3_copy = model.b3.clone()
        b_r_copy = model.b_r.clone()

        code_data1_z = torch.sigmoid(torch.matmul(z, W1_copy.t()) + b1_copy)
        code_data2_z = torch.sigmoid(torch.matmul(code_data1_z, W2_copy.t()) + b2_copy)
        #decode
        code_data3_z = torch.sigmoid(torch.matmul(code_data2_z, W2_copy) + b3_copy)
        recover_z = torch.matmul(code_data3_z, W1_copy) + b_r_copy

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
            Bx.append(b.cpu())
    Bx= torch.stack(Bx)
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