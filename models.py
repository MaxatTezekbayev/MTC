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
        self.W3 = nn.Parameter(torch.Tensor(code_sizes[0], code_sizes[1]))
        self.b3 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.W4 = nn.Parameter(torch.Tensor(dimensionality, code_sizes[0]))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W2, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W3, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W4, mean=0.0, std=1.0)
        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b2, 0.1)
        torch.nn.init.constant_(self.b3, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, calculate_jacobian = False, only_encode = False):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)

        if only_encode:
            return code_data2
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W3.t()) + self.b3)
        recover = torch.matmul(code_data3, self.W4.t()) + self.b_r

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
                grad_1 = torch.matmul(diag_sigma_prime1, self.W1)

                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(diag_sigma_prime2, self.W2)

                Jac.append(torch.matmul(grad_2, grad_1))
            Jac = torch.stack(Jac)
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
        self.W3 = nn.Parameter(torch.Tensor(code_sizes[0], code_sizes[1]))
        self.b3 = nn.Parameter(torch.Tensor(code_sizes[0]))
        self.W4 = nn.Parameter(torch.Tensor(dimensionality, code_sizes[0]))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W2, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W3, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.W4, mean=0.0, std=1.0)
        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b2, 0.1)
        torch.nn.init.constant_(self.b3, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, calculate_jacobian = False, calculate_DREI = False):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W3.t()) + self.b3)
        recover = torch.matmul(code_data3, self.W4.t()) + self.b_r

        if calculate_jacobian:
            Jac = []
            for i in range(x.shape[0]): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1[i], code_data1[i]))
                grad_1 = torch.matmul(diag_sigma_prime1, self.W1)

                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(diag_sigma_prime2, self.W2)

                diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data3[i], code_data3[i]))
                grad_3 = torch.matmul(diag_sigma_prime3, self.W3)
                grad_4 = self.W4

                Jac.append(torch.matmul(grad_4, torch.matmul(grad_3, torch.matmul(grad_2, grad_1))))
            Jac = torch.stack(Jac)
            return recover, code_data2, Jac

        if calculate_DREI:
            #drei
            A_matrix = []
            B_matrix = []
            C_matrix = []
            for i in range(x.shape[0]): 
                diag_sigma_prime1 = torch.diag( torch.mul(1.0 - code_data1[i], code_data1[i]))
                grad_1 = torch.matmul(diag_sigma_prime1, self.W1)

                diag_sigma_prime2 = torch.diag( torch.mul(1.0 - code_data2[i], code_data2[i]))
                grad_2 = torch.matmul(diag_sigma_prime2, self.W2)

                diag_sigma_prime3  = torch.diag( torch.mul(1.0 - code_data3[i], code_data3[i]))
                grad_3 = torch.matmul(diag_sigma_prime3, self.W3)
                    
                A_matrix.append(grad_1)
                B_matrix.append(grad_2)
                C_matrix.append(grad_3)
            A_matrix = torch.stack(A_matrix)
            B_matrix = torch.stack(B_matrix)
            C_matrix = torch.stack(CAE_matrix)
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
        code_data = self.CAE(x, only_encode = True)
        output = self.linear(code_data)
        return output