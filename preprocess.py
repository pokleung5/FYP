import utils
import torch

def preprocess_e(x):

    N = x.size()[-1]
    ex = utils.eignmatrix(x.view(-1, N, N)).view(-1, 1, N, N * 2)
    return ex.clone().detach().requires_grad_(True)

def preprocess_d(x):

    v = utils.vectorize_distance_from_DM(x)
    return v.clone().detach().requires_grad_(True)
    
def preprocess_m(x):

    return x.clone().detach().requires_grad_(True)

def preprocess_d_noise(x):

    noise = torch.randn(x.size()) * 0.1
    x = x + abs(noise) 
    x = utils.minmax_norm(x, dmin=0)[0]

    return preprocess_d(x)

# def preprocess_rnn(x):
    
#     N = x.size()[-1] 
#     x = x.view(-1, N, N)
#     x = x.permute(1, 0, 2)

#     return preprocess_m(x)