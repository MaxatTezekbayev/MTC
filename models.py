import torch
import torch.nn as nn


class CAE2Layer(nn.Module):
    def __init__(self, dimensionality, code_sizes):
        super(CAE2Layer, self).__init__()
        self.code_size=code_sizes[-1]
        # parameters
        self.W1 = nn.Parameter(torch.Tensor(code_sizes[0], dimensionality))
        self.b1 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.W2 = nn.Parameter(torch.Tensor(code_sizes[1], code_sizes[0]))
        self.b2 = nn.Parameter(torch.Tensor(code_sizes[1]))
        self.b3 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W2, mean=0.0, std=1.0)


        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b2, 0.1)
        torch.nn.init.constant_(self.b3, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, calculate_jacobian = False):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W2) + self.b3)
        recover = torch.matmul(code_data3, self.W1) + self.b_r

        batch_size = x.shape[0]
        #jacobian for CAEH is from encoded wrt input
        #autograd is slower
        #automatic:
            # grad_output=torch.ones(batch_size).cuda()
            # Jac=[]                                                                                        
            # for i in range(code_data2.shape[1]):
            #     Jac.append(torch.autograd.grad(outputs=code_data2[:,i], inputs=x, grad_outputs=grad_output, retain_graph=True, create_graph=True)[0])
            # Jac=torch.reshape(torch.cat(Jac,1),[x.shape[0], code_data2.shape[1], x.shape[1]])
    
        #https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
        if calculate_jacobian:
            Jac = []
            for i in range(batch_size): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1[i], code_data1[i]))
                grad_1 = torch.matmul(self.W1.t(), diag_sigma_prime1)
    
                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(self.W2.t(), diag_sigma_prime2)
        
                Jac.append(torch.matmul(grad_1, grad_2))
            Jac = torch.reshape(torch.cat(Jac,1),[batch_size, code_data2.shape[1], x.shape[1]])
            return recover, code_data2, Jac
        return recover,  code_data2, 

class ALTER2Layer(nn.Module):
    def __init__(self, dimensionality, code_sizes):
        super(ALTER2Layer, self).__init__()
        self.code_size=code_sizes[-1]
        # parameters
        self.W1 = nn.Parameter(torch.Tensor(code_sizes[0], dimensionality))
        self.b1 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.W2 = nn.Parameter(torch.Tensor(code_sizes[1], code_sizes[0]))
        self.b2 = nn.Parameter(torch.Tensor(code_sizes[1]))
        self.b3 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W2, mean=0.0, std=1.0)

        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b2, 0.1)
        torch.nn.init.constant_(self.b3, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, calculate_jacobian = False, calculate_DREI = False):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W2) + self.b3)
        recover = torch.matmul(code_data3, self.W1) + self.b_r

        if calculate_jacobian:
            Jac = []
            for i in range(x.shape[0]): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1[i], code_data1[i]))
                grad_1 = torch.matmul(self.W1.t(), diag_sigma_prime1)
    
                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(self.W2.t(), diag_sigma_prime2)
                
                diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data3[i], code_data3[i]))
                grad_3 = torch.matmul(self.W2, diag_sigma_prime3)

                grad_4 = self.W1
                Jac.append(torch.matmul(grad_1, torch.matmul(grad_2, torch.matmul(grad_3, grad_4))))
            Jac = torch.reshape(torch.cat(Jac,1), [x.shape[0], recover.shape[1], x.shape[1]])
            return recover, code_data2, Jac

        if calculate_DREI:
            #drei
            A_matrix = []
            B_matrix = []
            C_matrix = []
            for i in range(x.shape[0]): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1[i], code_data1[i]))
                grad_1 = torch.matmul(W1_copy.t(), diag_sigma_prime1)

                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(W2_copy.t(), diag_sigma_prime2)
        
                diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data3[i], code_data3[i]))
                grad_3 = torch.matmul(W2_copy, diag_sigma_prime3)
        
                A_matrix.append(grad_1)
                B_matrix.append(grad_2)
                C_matrix.append(grad_3)
            A_matrix = torch.reshape(torch.cat(A_matrix, 1),[x.shape[0], grad_1.shape[0], grad_1.shape[1]])
            B_matrix = torch.reshape(torch.cat(B_matrix, 1),[x.shape[0], grad_2.shape[0], grad_2.shape[1]])
            C_matrix = torch.reshape(torch.cat(C_matrix, 1),[x.shape[0], grad_3.shape[0], grad_3.shape[1]])
            return recover, code_data2, A_matrix, B_matrix, C_matrix
        return recover, code_data2
        
        
class MTC(nn.Module):
    def __init__(self, CAE_model, output_dim):
        super(MTC, self).__init__()
        self.CAE = CAE_model
        self.output_dim = output_dim
        # parameters

        self.linear= nn.Linear(self.CAE.code_size, output_dim) 


    def forward(self, x):
        #encode
        recover, code_data = self.CAE(x)
        output = self.linear(code_data)

        return output