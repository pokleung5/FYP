
import json
from torch.utils.data import DataLoader
import torch
from torch import nn, Tensor
import pandas
import numpy
import utils

torch.set_default_tensor_type('torch.DoubleTensor')

# %%


class TrainHelper:

    def __init__(self, id, model, optimizer, lossFun, lr_factor=0.1):

        self.id = id
        self.model = model
        self.optim = optimizer
        self.lossFun = lossFun

        if lr_factor is None:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=lr_factor, verbose=False)

        self._init_record()

    def _init_record(self):

        self.epoch = 0
        self.records = pandas.DataFrame(
            columns=['lr', 'loss_mean', 'loss_max', 'loss_min', 'train_time'])
        self.localRecord = []

    def _add_record_to_local(self, loss_values, time_cost):

        r = [self.optim.param_groups[0]['lr'],
             numpy.mean(loss_values), numpy.max(
                 loss_values), numpy.min(loss_values),
             numpy.sum(time_cost)]

        self.localRecord.append(dict(zip(self.records.columns, r)))

    def _train(self, data: Tensor):
        self.optim.zero_grad()

        rs = self.model(data)
        loss = self.lossFun(rs, data.data)

        if loss != loss:
            raise Exception("Gradient Disconnected !")

        loss.backward()
        self.optim.step()

        return loss

    def plot(self, cols=None, value_label="Loss"):
        if cols:
            utils.plot_records(
                self.records[cols].to_dict(orient='list'),
                self.epoch, value_label=value_label)
        else:
            utils.plot_records(
                self.records.to_dict(orient='list'),
                self.epoch, value_label=value_label)

    def backup(self):
        utils.dump_variable(self, 'backup/' + self.id + '_' + str(self.epoch) + '.model')

    def merge_local_record(self):

        self.epoch += len(self.localRecord)
        self.records = self.records.append(self.localRecord)
        self.localRecord = []

    def step_scheduler(self, x):

        self.scheduler.step(x)

    def train(self, dlr: DataLoader, EPOCH: int, print_on_each=0, append_record=True):

        if not append_record:
            self._init_record()

        if len(self.localRecord) > 0:
            raise Exception("Record in last train not merged !")

        for epoch in range(EPOCH):

            loss_values, time_cost = [], []

            for step, data in enumerate(dlr):

                loss, t = utils.time_measure(self._train, [data])

                loss_values.append(loss.data)
                time_cost.append(t)

            self._add_record_to_local(loss_values, time_cost)

            if print_on_each > 0 and epoch % print_on_each == 0:
                print(epoch, "\t| Mean loss:",
                      self.localRecord[epoch]['loss_mean'])

            if self.scheduler is not None:
                self.step_scheduler(self.localRecord[epoch]['loss_mean'])

        self.merge_local_record()

    def output_state(self):

        print("MODEL")
        print(self.model.state_dict())
        print("OPTIM")
        print(self.optim.state_dict())

# %%
