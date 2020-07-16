
import json
from torch.utils.data import DataLoader
import torch
from torch import nn, Tensor
import pandas
import numpy
import utils

torch.set_default_tensor_type('torch.DoubleTensor')

# %%
# for filepath in glob.iglob('backup/*.model'):
#     h = utils.load_variable(filepath)
#     a = trainHelper.TrainHelper(h.id, h.model, h.optim, h.lossFun, h.preprocess)

#     a.scheduler = h.scheduler
#     a.records = h.records
#     a.epoch = h.epoch
#     a.localRecord = h.localRecord

#     a.backup()
# %%


class TrainHelper:

    def __init__(self, id, model, optimizer, lossFun, preprocess=None, lr_factor=0.1):

        self.id = id
        self.model = model
        self.optim = optimizer
        self.lossFun = lossFun

        self.preprocess = preprocess

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=lr_factor, verbose=False)

        self.__init_record()
        self.config_fit()

    # def __init__(self, helper):

    #     self.id = helper.id
    #     self.model = helper.model
    #     self.optim = helper.optim
    #     self.lossFun = helper.lossFun
    #     self.preprocess = helper.preprocess
    #     self.scheduler = helper.scheduler
    #     self.epoch = helper.epoch
    #     self.records = helper.records
    #     self.localRecord = helper.localRecord

    def __init_record(self):

        self.epoch = 0
        self.records = pandas.DataFrame(
            columns=['lr', 'loss_mean', 'loss_max', 'loss_min', 'train_time'])
        self.localRecord = []

    def __add_record_to_local(self, loss_values, time_cost):

        r = [self.optim.param_groups[0]['lr'],
             numpy.mean(loss_values), numpy.max(
                 loss_values), numpy.min(loss_values),
             numpy.sum(time_cost)]

        self.localRecord.append(dict(zip(self.records.columns, r)))

    def __train(self, data: Tensor):

        self.optim.zero_grad()

        rs, target = self.__predict(data)

        loss = self.lossFun(rs, target.data)

        if loss != loss:
            raise Exception("Gradient Disconnected !")

        loss.backward()
        self.optim.step()

        return loss

    def __predict(self, data):

        target = data

        if self.preprocess is not None:
            data, target = self.preprocess(data)

        return self.model(data), target


    def predict(self, data):

        if self.epoch == 0:
            self.quick_fit(data)

        target = data

        if self.preprocess is not None:
            data, target = self.preprocess(data)

        return self.model.encode(data), target

    def backup(self):

        utils.dump_variable(self, 'backup/' + self.id +
                            '_' + str(self.epoch) + '.model')

    def merge_local_record(self):

        self.epoch += len(self.localRecord)
        self.records = self.records.append(self.localRecord)
        self.localRecord = []

    def step_scheduler(self, x):

        self.scheduler.step(x)

    def config_fit(self, minlr=1e-8, minEpoch=0, maxEpoch=50):

        self.fit_minlr = minlr
        self.fit_minEpoch = minEpoch
        self.fit_maxEpoch = maxEpoch 
        

    def quick_fit(self, x):
            
        self.epoch = 0

        while self.epoch < self.fit_minEpoch or (
            self.optim.param_groups[0]['lr'] > self.fit_minlr and self.epoch < self.fit_maxEpoch):

            self.epoch = self.epoch + 1
            self.__train(x)
            

    def train(self, dlr: DataLoader, EPOCH: int):
        
        if len(self.localRecord) > 0:
            raise Exception("Record in last train not merged !")

        for epoch in range(EPOCH):

            loss_values, time_cost = [], []

            for step, data in enumerate(dlr):

                loss, t = utils.time_measure(self.__train, [data])

                loss_values.append(loss.data)
                time_cost.append(t)

            self.__add_record_to_local(loss_values, time_cost)
            self.step_scheduler(self.localRecord[epoch]['loss_mean'])

        self.merge_local_record()


        
        




# %%
