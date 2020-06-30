from train import *

# sample_space=(1000, 1)
data = dataSource.load_data(shape=(ss, N, d))[0]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3
lossFun = CoordsToDMLoss(N, 2, lossFun=nn.MSELoss(reduction='sum'))

for neuron in range(800, 89, 16):

    for i in range(1, 6):

        model_id = "Coord_Linear_" + str(i)+ "_" + str(neuron) + "_E_MSE"

        in_dim, out_dim = N * 2, 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_e, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_linear2.log')

        ###########################################################################

        model_id = "Coord_Linear_" + str(i)+ "_" + str(neuron) + "_D_MSE"

        in_dim, out_dim = int((N * N - N) / 2), N * 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        
        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_d, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_linear2.log')
        
        ###########################################################################

        model_id = "Coord_Linear_" + str(i)+ "_" + str(neuron) + "_M_MSE"

        in_dim, out_dim = N, 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2), out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        
        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_m, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_linear2.log')
