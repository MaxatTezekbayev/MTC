import torch
import torch.nn as nn

class CAE1Layer(nn.Module):
    def __init__(self, dimensionality, code_size):
        super(CAE1Layer, self).__init__()
        # parameters
        self.W1 = nn.Parameter(torch.Tensor(code_size, dimensionality))
        self.b1 = nn.Parameter(torch.Tensor(code_size))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()
        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)

    def forward(self, x, x_noise):
        code_data = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        recover = self.sigmoid(torch.matmul(code_data, self.W1) + self.b_r)

        #noise
        code_data_noise = torch.sigmoid(torch.matmul(x_noise, self.W1.t()) + self.b1)
        
        return recover, code_data, code_data_noise


class CAE2Layer(nn.Module):
    def __init__(self, dimensionality, code_sizes):
        super(CAE2Layer, self).__init__()
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

    def forward(self, x, x_noise):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W2) + self.b3)
        recover = self.sigmoid(torch.matmul(code_data3, self.W1) + self.b_r)


        code_data_noise1 = torch.sigmoid(torch.matmul(x_noise, self.W1.t()) + self.b1)
        code_data_noise2 = torch.sigmoid(torch.matmul(code_data_noise1, self.W2.t()) + self.b2)

        return recover, code_data2, code_data_noise2


# class MCT(nn.Module):
#     def __init__(self, CAE_model, output_dim):
#         super(CAE2Layer, self).__init__()
#         self.CAE = CAE_model
#         self.output_dim = output_dim
#         # parameters
#         #encoder
#         self.softmax = nn.Parameter(torch.Tensor(self.dim, code_sizes[0]))
#         self.b1 = nn.Parameter(torch.Tensor(code_sizes[0]))
#         self.W2 = nn.Parameter(torch.Tensor(code_sizes[0], code_sizes[1]))
#         self.b2 = nn.Parameter(torch.Tensor(code_sizes[1]))
#         #decoder
#         self.W3 = nn.Parameter(torch.Tensor(code_sizes[1], code_sizes[0]))
#         self.b3 = nn.Parameter(torch.Tensor(code_sizes[0]))
#         self.W4 = nn.Parameter(torch.Tensor(code_sizes[0], self.dim))
#         self.b_r = nn.Parameter(torch.Tensor(self.dim))

#         self.sigmoid = torch.nn.Sigmoid()
#         # init
#         torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
#         torch.nn.init.normal_(self.W2, mean=0.0, std=1.0)
#         torch.nn.init.normal_(self.W3, mean=0.0, std=1.0)
#         torch.nn.init.normal_(self.W4, mean=0.0, std=1.0)

#         torch.nn.init.constant_(self.b1, 0.1)
#         torch.nn.init.constant_(self.b2, 0.1)
#         torch.nn.init.constant_(self.b3, 0.1)
#         torch.nn.init.constant_(self.b_r, 0.1)

#     def forward(self, x, x_noise):
#         #encode
#         code_data1 = self.sigmoid(torch.matmul(x.view(-1, self.dim), self.W1) + self.b1)
#         code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2) + self.b2)
#         #decode
#         code_data3 = self.sigmoid(torch.matmul(code_data2, self.W3) + self.b3)
        
#         recover = self.sigmoid(torch.matmul(code_data3, self.W4) + self.b_r)
#         recover = recover.view(*x.shape)


#         code_data_noise1 = torch.sigmoid(torch.matmul(x_noise.view(-1, self.Nnoise, self.dim), self.W1) + self.b1)
#         code_data_noise2 = torch.sigmoid(torch.matmul(code_data_noise1, self.W2) + self.b2)

#         return recover, code_data2, code_data_noise2