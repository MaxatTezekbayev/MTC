import torch
import torch.nn as nn

class CAE1Layer(nn.Module):
    def __init__(self, dimensionality, code_size):
        super(CAE1Layer, self).__init__()
        self.code_size=code_size
        # parameters
        self.W1 = nn.Parameter(torch.Tensor(code_size, dimensionality))
        self.b1 = nn.Parameter(torch.Tensor(code_size))
        self.b_r = nn.Parameter(torch.Tensor(dimensionality))

        self.sigmoid = torch.nn.Sigmoid()

        # init
        torch.nn.init.normal_(self.W1, mean=0.0, std=1.0)
        torch.nn.init.constant_(self.b1, 0.1)
        torch.nn.init.constant_(self.b_r, 0.1)



    def forward(self, x, x_noise=None):
        code_data = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        recover = self.sigmoid(torch.matmul(code_data, self.W1) + self.b_r)

        #noise
        if x_noise is not None:
            code_data_noise = torch.sigmoid(torch.matmul(x_noise, self.W1.t()) + self.b1)
            return recover, code_data, code_data_noise
        else:
            return recover, code_data

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

    def forward(self, x, x_noise=None):
        #encode
        code_data1 = self.sigmoid(torch.matmul(x, self.W1.t()) + self.b1)
        code_data2 = self.sigmoid(torch.matmul(code_data1, self.W2.t()) + self.b2)
        #decode
        code_data3 = self.sigmoid(torch.matmul(code_data2, self.W2) + self.b3)
        recover = self.sigmoid(torch.matmul(code_data3, self.W1) + self.b_r)

        if x_noise is not None:
            code_data_noise1 = torch.sigmoid(torch.matmul(x_noise, self.W1.t()) + self.b1)
            code_data_noise2 = torch.sigmoid(torch.matmul(code_data_noise1, self.W2.t()) + self.b2)
            return recover, code_data2, code_data_noise2
        else:
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