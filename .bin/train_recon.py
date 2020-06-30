from train import *

data = dataSource.load_data(shape=(ss, N, d))[1]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3
lossFun = ReconLoss(lossFun=nn.MSELoss(reduction='sum'))

for neuron in range(8, 89, 16):

    for i in range(1, 6):

        model_id = "Recon_Linear_" + str(i)+ "_" + str(neuron) + "_E_MSE"

        in_dim = N * 2
        out_dim = N

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_e, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_recon.log')

        ###########################################################################

        model_id = "Recon_AE_" + str(i)+ "_" + str(neuron) + "_E_MSE"

        in_dim = int(N * (N - 1) / 2)
        out_dim = in_dim

        model = ae.AutoEncoder([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_e, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_recon.log')

