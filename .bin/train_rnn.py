from train import *

# %%
# sample_space=(1000, 1)
data = dataSource.load_data(shape=(ss, N, d))[0]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3
lossFun = CoordsToDMLoss(N, 2, lossFun=nn.MSELoss(reduction='sum'))

for neuron in range(8, 89, 16):

    for i in range(1, 4):

        model_id = "Coords_2aRNN_" + str(i)+ "_" + str(neuron) + "_M_MSE"

        in_dim, out_dim = N, 2

        model = rnn.aRNN(N, [*[neuron for j in range(i - 1)], int(neuron / 2),  out_dim], num_rnn_layers=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_rnn, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_rnn.log')

        ###########################################################################

        model_id = "Coords_2bRNN_" + str(i)+ "_" + str(neuron) + "_M_MSE"

        in_dim, out_dim = N, 2

        model = rnn.bRNN(N, [in_dim, *[neuron for j in range(i - 1)], int(neuron / 2)], out_dim, num_rnn_layers=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_rnn, lossFun=lossFun)
        
        train(helper, dlr, 'log/train_rnn2.log')
