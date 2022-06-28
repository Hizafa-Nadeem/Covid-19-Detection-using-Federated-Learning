import torch
import torch.nn as nn
from fastai.vision.core import ConvLayer, Flatten
import numpy as np
import torch.optim as optim
import dill


def conv_block(ni, nf, size=3, stride=1):
    for_pad = lambda s: s if s > 2 else 3
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=size, stride=stride, padding=(for_pad(size) - 1) // 2, bias=False),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )


def triple_conv(ni, nf):
    return nn.Sequential(
        conv_block(ni, nf),
        conv_block(nf, ni, size=1),
        conv_block(ni, nf)
    )


def maxpooling():
    return nn.MaxPool2d(2, stride=2)


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 8),
            maxpooling(),
            conv_block(8, 16),
            maxpooling(),
            triple_conv(16, 32),
            maxpooling(),
            triple_conv(32, 64),
            maxpooling(),
            triple_conv(64, 128),
            maxpooling(),
            triple_conv(128, 256),
            conv_block(256, 128, size=1),
            conv_block(128, 256),
            ConvLayer(256, 2),  # conv_
            Flatten(),
            nn.Linear(338, 2),
            # nn.Sigmoid(),
            # Flatten(),
            # nn.Softmax(dim=1)   # get value between 0 and 1
        )

    def forward(self, x):
        z = self.model(x)
        return z



class AdamOptimizer:
    def __init__(self, num_items,num_factors, beta_1=0.9, beta_2=0.99, gemma=0.003, epsilon=1e-8):
        """Initialize
        Args:
        num_items:  Total number of items in the data
        num_factors: Number of factors
        beta_1: hyper-parameter
        beta_2: hyper-parameter
        gemma: hyper-parameter
        epsilon: hyper-parameter
        """
        self.num_items = num_items
        self.num_factors = num_factors
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.gemma = gemma
        self.epsilon = epsilon

        self.m = np.zeros((self.num_items, self.num_factors))
        self.v = np.zeros((self.num_items, self.num_factors))
        self.t = 0

    def dump_object(self):
        serialized_object = None
        try:
            serialized_object = dill.dumps({
                'num_items': self.num_items,
                'num_factors': self.num_factors,
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'gemma': self.gemma,
                'epsilon': self.epsilon,
                'm': self.m,
                'v': self.v,
                't': self.t
            })
        except:
            print("Error Serializing --dill, AdamOptimizer")
        return serialized_object

    def load_object(self, obj):
        u_obj = None  # Un-serializing object
        try:
            u_obj = dill.loads(obj)
            self.num_items = u_obj['num_items']
            self.num_factors = u_obj['num_factors']
            self.beta_1 = u_obj['beta_1']
            self.beta_2 = u_obj['beta_2']
            self.gemma = u_obj['gemma']
            self.epsilon = u_obj['epsilon']
            self.m = u_obj['m']
            self.v = u_obj['v']
            self.t = u_obj['t']
        except:
            print("Error Un-serializing --dill, AdamOptimizer")
        return u_obj

    # This method is used to calculate the gradient using Adam optimizer
    def optimize_gradients(self, gradients):

        self.t += 1

        self.m = (self.beta_1 * self.m) + (1.0 - self.beta_1) * gradients  # Equation 13

        self.v = (self.beta_2 * self.v) + (1.0 - self.beta_2) * np.square(gradients)  # Equation 14

        # Calculates the bias-corrected estimates
        m_hat = self.m / (1.0 - (self.beta_1 ** self.t))  # Equation 15
        v_hat = self.v / (1.0 - (self.beta_2 ** self.t))  # Equation 16

        new_gradients = self.gemma * (m_hat) / (np.sqrt(v_hat) + self.epsilon)  # Equation 17

        return new_gradients

class FLServer():
    """
    # Receive and load gradients from FL_Client
    # Update weights using gradients
    # Send weights to FL_Server
    """

    def __init__(self):
        self.ae = AE()
        #self.ae = model2
        #self.ae = torch.load("model1.pt")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.ae.parameters(), lr = 2e-5)
        self.dct = {} #layer:layer.shape
        self.adam_optimizer = {} #layer:optim_for_that_layer
        self.new_grad = []

        for layer, param in self.ae.named_parameters():
          #if param.grad is not None:
          if param.requires_grad==True:
            self.dct[layer] = param.shape
            #param.data = torch.zeros(param.shape)
            shp = param.data.reshape(param.shape[0],-1).shape
            # print(param.shape," -> ",shp)
            self.adam_optimizer[layer] = AdamOptimizer(shp[0], shp[1])


    def save_model(self):  # save neural network on filesystem
        torch.save(self.ae.state_dict(), "ae.pt")
        print("Model Saved")

    def load_model(self):  # load neural network from filesystem
        self.ae.load_state_dict(torch.load("ae.pt"))
        # self.ae.eval()
        # self.ae.train()
        self.optimizer = optim.Adam(self.ae.parameters(), lr=3e-3)
        print("Model Loaded")

    def dump_object(self):  # store adam_optimizer and dct in redis
        s_obj = None
        lst = {}
        for key in self.adam_optimizer:
            lst[key] = self.adam_optimizer[key].dump_object()

        lst2 = {}
        for key in self.dct:
            lst2[key] = self.dct[key]

        try:
            s_obj = dill.dumps({
                'adam_optimizer': lst,
                'dct': lst2
            })
        except:
            print("Error serializing FLServer")
        return s_obj

    def load_object(self, s_obj):
        u_obj = None  # Un-serializing object
        try:
            u_obj = dill.loads(s_obj)
            lst = u_obj['adam_optimizer']
            lst2 = u_obj['dct']

            for key in lst:
                self.adam_optimizer[key].load_object(lst[key])
            for key in lst2:
                self.dct[key] = lst2[key]

        except:
            print("Error Un-serializing --dill, FLServer")
        return u_obj

    def update_grad(self, new_grad):

        i = -1
        for layer, param in self.ae.named_parameters():
            if param.requires_grad == True:
                i += 1
                new_grad[i] = self.adam_optimizer[layer].optimize_gradients(new_grad[i].reshape(param.shape[0], -1))

        with torch.no_grad():
            self.optimizer.zero_grad()
            i = -1
            for layer, param in self.ae.named_parameters():
                if param.requires_grad == True:
                    i += 1
                    # param.grad = new_grad[i].reshape(self.dct[layer])
                    param.grad = torch.Tensor(new_grad[i].reshape(self.dct[layer]))


    def save_model(self):  # save neural network on filesystem
        torch.save(self.ae.state_dict(), "ae.pt")
        print("Model Saved")

    def train_model(self):
      ct=0
      with torch.no_grad():
        for layer, param in self.ae.named_parameters():
          #if param.grad is not None:
          if param.requires_grad==True:
            ct=ct+1
            param.data -= param.grad
      print("parameters trained: " + str(ct))

    def get_weights(self):
      with torch.no_grad():
        weights = []
        for name, param in self.ae.named_parameters():
          weights.append(param.data)
        return weights


# class FLServer:
#
#
#     def __init__(self):
#         self.ae = torch.load("model1.pt")
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.ae.parameters(), lr=3e-3)
#         self.dct = {}  # layer:layer.shape
#         self.adam_optimizer = {}  # layer:optim_for_that_layer
#
#         for layer, param in self.ae.named_parameters():
#             if param.requires_grad == True:
#                 self.dct[layer] = param.shape
#                 # param.data = torch.zeros(param.shape)
#                 shp = param.data.reshape(param.shape[0], -1).shape
#                 # print(param.shape," -> ",shp)
#                 self.adam_optimizer[layer] = AdamOptimizer(shp[0], shp[1])
#
#     def save_model(self):  # save neural network on filesystem
#         torch.save(self.ae.state_dict(), "ae.pt")
#         print("Model Saved")
#
#     def load_model(self):  # load neural network from filesystem
#         self.ae.load_state_dict(torch.load("ae.pt"))
#         # self.ae.eval()
#         # self.ae.train()
#         self.optimizer = optim.Adam(self.ae.parameters(), lr=3e-3)
#         print("Model Loaded")
#
#     def dump_object(self):  # store adam_optimizer and dct in redis
#         s_obj = None
#         lst = {}
#         for key in self.adam_optimizer:
#             lst[key] = self.adam_optimizer[key].dump_object()
#
#         lst2 = {}
#         for key in self.dct:
#             lst2[key] = self.dct[key]
#
#         try:
#             s_obj = dill.dumps({
#                 'adam_optimizer': lst,
#                 'dct': lst2
#             })
#         except:
#             print("Error serializing FLServer")
#         return s_obj
#
#     def load_object(self, s_obj):
#         u_obj = None  # Un-serializing object
#         try:
#             u_obj = dill.loads(s_obj)
#             lst = u_obj['adam_optimizer']
#             lst2 = u_obj['dct']
#
#             for key in lst:
#                 self.adam_optimizer[key].load_object(lst[key])
#             for key in lst2:
#                 self.dct[key] = lst2[key]
#
#         except:
#             print("Error Un-serializing --dill, FLServer")
#         return u_obj
#
#     def update_grad(self, new_grad):
#
#         i = -1
#         for layer, param in self.ae.named_parameters():
#             if param.requires_grad == True:
#                 i += 1
#                 new_grad[i] = self.adam_optimizer[layer].optimize_gradients(new_grad[i].reshape(param.shape[0], -1))
#
#         with torch.no_grad():
#             self.optimizer.zero_grad()
#             i = -1
#             for layer, param in self.ae.named_parameters():
#                 if param.requires_grad == True:
#                     i += 1
#                     # param.grad = new_grad[i].reshape(self.dct[layer])
#                     param.grad = torch.Tensor(new_grad[i].reshape(self.dct[layer]))
#
#     def old_update_grad(self, grad_arr):
#
#         new_grad = [np.zeros_like(x) for x in grad_arr[0]]
#
#         for gradient in grad_arr:
#             for i in range(len(gradient)):
#                 new_grad[i] += np.array(gradient[i])
#
#         i = -1
#         for layer, param in self.ae.named_parameters():
#             i += 1
#             new_grad[i] = self.adam_optimizer[layer].optimize_gradients(new_grad[i].reshape(param.shape[0], -1))
#
#         with torch.no_grad():
#             self.optimizer.zero_grad()
#             i = -1
#             for layer, param in self.ae.named_parameters():
#                 i += 1
#                 # param.grad = new_grad[i].reshape(self.dct[layer])
#                 param.grad = torch.Tensor(new_grad[i].reshape(self.dct[layer]))
#
#     def train_model(self):
#         with torch.no_grad():
#             for layer, param in self.ae.named_parameters():
#                 if param.grad is not None:
#                     param.data -= param.grad
#
#     def train_model_v2(self):
#         self.optimizer.step()
#
#     def get_weights(self):
#         with torch.no_grad():
#             weights = []
#             for name, param in self.ae.named_parameters():
#                 weights.append(param.data)
#             return weights
