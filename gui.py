import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tkinter import *

import numpy
import torch

import utils
import dataSource
import manifold

import matplotlib.cm as cm

matplotlib.use('TkAgg')
torch.set_default_tensor_type('torch.DoubleTensor')

font = {'size': 7}
matplotlib.rc('font', **font)

color = ['g', 'y', 'r', 'c', 'm']


class Application:

    def __init__(self, N, d, master=None):

        self.N = N
        self.d = d

        self.master = master

        self.etm = manifold.Algorithm(N, d)
        # self.etm.use_pretrained_model('result\Coord_StepLinear_E2_3MSE_5_82_400.model')

        self.methods = {'Classical MDS':    self.etm.classicalMDS,
                        'Landmark MDS':     self.etm.landmarkMDS,
                        'Non-Metric MDS':   self.etm.nonMetricMDS,
                        'Fastmap':          self.etm.fastmap,
                        'Deep MDS':          self.etm.deepMDS
                        }

        self.methods_name = [*self.methods.keys()]
        self.methods_color = dict(
            zip(self.methods_name, color[:len(self.methods_name)]))
        self.methods_color['True Data'] = 'b'

        self.dot_list = {}
        self.anno_list = {}

        self.colorized = False
        self.showingTrueLabel = False

        self._buildDmPane().pack(side=LEFT, expand=None, fill=BOTH, padx=(20, 0))
        self._buildCoordPane().pack(side=TOP, expand=Y, fill=BOTH,
                                    padx=(10, 20), pady=(30, 0))
        self._buildPlotPane().pack(side=BOTTOM, expand=YES, fill=BOTH)

        self.btn_genDM_fun()

    def _buildPlotPane(self):

        fig = plt.figure(figsize=(5, 5))
        self.ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        plotPane = Frame(self.master)
        plotPane.config(background='#FFFFFF')

        plotPane.grid_columnconfigure((0, 5), weight=1)
        plotPane.grid_rowconfigure((0, 5), weight=1)

        self.canvas = FigureCanvasTkAgg(fig, master=plotPane)
        self.canvas.get_tk_widget().grid(row=1, column=1, columnspan=4, rowspan=3)
        self.canvas.draw()

        btn_removeLabel = Button(
            plotPane, text='Clear Current Label', command=self._plotRemove)
        btn_removeLabel.grid(
            row=4, column=1, columnspan=1, pady=10, padx=(30, 0), sticky='nesw')

        btn_removeLabel = Button(
            plotPane, text='Clear All Label', command=self._plotClear)
        btn_removeLabel.grid(
            row=4, column=2, columnspan=1, pady=10, padx=(10, 10), sticky='nesw')

        self.btn_toggleTrueLabel = Button(plotPane, text='Hide True Label',
                                          command=self._toggleTrueLabel)
        self.btn_toggleTrueLabel .grid(
            row=4, column=3, columnspan=1, pady=10, padx=(0, 30), sticky='nesw')

        return plotPane

    def _buildCoordPane(self):

        self.rsCoordViews = []

        gridx, gridy = (self.d + 2, self.N + 2)

        coordPane = Frame(self.master, background='#FFFFFF')

        coordPane.grid_columnconfigure((0, gridy + 1), weight=1)
        coordPane.grid_rowconfigure((0, gridx + 1), weight=1)

        Label(coordPane, width=6, text='X').grid(
            row=2, column=1, padx=5, pady=2, sticky='nesw')

        Label(coordPane, width=6, text='Y').grid(
            row=3, column=1, padx=5, pady=2, sticky='nesw')

        for i in range(2, gridy):

            Label(coordPane, width=6, text=str(i - 2)).grid(
                row=1, column=i, padx=5, pady=2, sticky='nesw')

            for j in range(2, gridx):

                t = Entry(coordPane, width=6, justify=RIGHT)
                t.grid(row=j, column=i, padx=5, pady=2, sticky='nesw')

                self.rsCoordViews.append(t)

        self.opt_method = StringVar(self.master)
        self.opt_method.set('Select a method :')

        meun_method = OptionMenu(
            coordPane, self.opt_method, *self.methods_name, command=self.btn_rsCoord_fun)
        meun_method.grid(
            row=gridy, column=1, columnspan=gridy, pady=10, padx=10, sticky='ew')
        meun_method.config(width=20)

        return coordPane

    def _buildDmPane(self):

        self.genDmViews = []
        self.rsDmViews = []

        gridx, gridy = (self.d + 2, self.N + 2)

        dmPane = Frame(self.master, background='#FFFFFF')  # , height=200)

        dmPane_gen = Frame(dmPane, background='#FFFFFF')  # , height=200
        dmPane_gen.pack(side=TOP, expand=YES, fill=None, padx=5)
        dmPane_gen.grid_columnconfigure((0, gridy + 1), weight=1)

        dmPane_rs = Frame(dmPane, background='#FFFFFF')  # , height=200
        dmPane_rs.pack(side=LEFT, expand=YES, fill=X, padx=5, pady=(0, 20))
        dmPane_rs.grid_columnconfigure((0, gridy), weight=1)
        # dmPane_rs.config(highlightbackground="black", highlightthickness=1)

        for i in range(2, gridy):

            Label(dmPane_gen, width=5, text=str(i - 2)).grid(
                row=1, column=i, padx=5, pady=2, sticky='nesw')

            Label(dmPane_gen, width=5, text=str(i - 2)).grid(
                row=i, column=1, padx=5, pady=2, sticky='nesw')

            Label(dmPane_rs, width=5, text=str(i - 2)).grid(
                row=1, column=i, padx=5, pady=2, sticky='nesw')

            Label(dmPane_rs, width=5, text=str(i - 2)).grid(
                row=i, column=1, padx=5, pady=2, sticky='nesw')

            for j in range(2, gridy):

                t = Entry(dmPane_gen, width=5, justify=RIGHT)
                t.grid(row=i, column=j, padx=5, pady=2, sticky='nesw')

                k = Entry(dmPane_rs, width=5, justify=RIGHT)
                k.grid(row=i, column=j, padx=5, pady=2, sticky='nesw')

                self.genDmViews.append(t)
                self.rsDmViews.append(k)

        btn_genEuDM = Button(
            dmPane_gen, text='Generate Euclidean Data', command=self.btn_genEuDM)
        btn_genEuDM.grid(
            row=gridy, column=0, columnspan=gridy + 1, pady=(10, 0), padx=30, sticky='nesw')

        btn_genDM = Button(
            dmPane_gen, text='Generate Random Data', command=self.btn_genDM_fun)
        btn_genDM.grid(
            row=gridy + 1, column=0, columnspan=gridy + 1, pady=(0, 10), padx=30, sticky='nesw')

        self.label_loss = Label(dmPane_rs, background='#b8b8b8')
        self.label_loss.grid(
            row=0, column=0, columnspan=gridy - 3, pady=10, padx=(30, 10), sticky='nesw')

        btn_colorize = Button(dmPane_rs, text='Colorize Distance',
                              command=lambda: self._colorize_diff(self.rsDmViews))
        btn_colorize.grid(
            row=0, column=gridy - 3, columnspan=3, pady=10, padx=(0, 30), sticky='nesw')

        return dmPane

    def btn_rsCoord_fun(self, method_name=None):

        dm = self._getEntry(self.genDmViews).reshape(self.N, self.N)

        if method_name is None:
            method_name = self.opt_method.get()

        if method_name == 'Deep MDS':
            
            model_path = filedialog.askopenfilename(
                initialdir=".", title="Select A Model", filetypes=(("Pretained Model", "*.model"), ("all files", "*.*")))
            
            if model_path == '':
                return

            self.opt_method.set(model_path.split('\\')[-1].split('/')[-1])
            self.etm.use_pretrained_model(model_path)
            
            self._plotRemove(method_name)

        data = self.methods[method_name](dm)
        # try:
        #     data = self.methods[method_name](dm)
        # except:
        #     self.opt_method.set(method_name + ' failed to get result !')
        #     self._fillEntry(data=numpy.zeros(self.N * self.d),
        #                     entrylist=self.rsCoordViews)
        #     return

        data = torch.tensor(data)
        data = utils.minmax_norm(data.view(1, -1))[0]

        self._fillEntry(data=data, entrylist=self.rsCoordViews)

        coords = data.reshape(self.N, self.d)
        self._plotData(x=coords[:, 0], y=coords[:, 1], label=method_name)

        dm = torch.tensor(dm)

        rs_dm = utils.get_distanceSq_matrix(coords) ** 0.5
        rs_dm = utils.minmax_norm(rs_dm)[0] * torch.max(dm)

        loss = torch.mean(torch.pow(dm - rs_dm.view_as(dm), 2))

        self.label_loss.config(text='MSE = ' + str(float(loss)))

        self._fillEntry(data=rs_dm, entrylist=self.rsDmViews)

        self.colorized = not self.colorized
        self._colorize_diff(self.rsDmViews)

    def btn_genEuDM(self):

        self._plotClear()

        self.coords = dataSource.get_rand_data((1, self.N, 2), isInt=True,
                                               maxXY=maxXY, minXY=minXY)
        self.coords = utils.minmax_norm(self.coords)[0]

        data = utils.get_distanceSq_matrix(self.coords) ** 0.5
        data = utils.minmax_norm(data)[0]

        self._fillEntry(data=data, entrylist=self.genDmViews)

        self.showingTrueLabel = False
        self._toggleTrueLabel()

    def btn_genDM_fun(self):

        self._plotClear()

        self.coords = None

        data = dataSource.generate_rand_DM(
            N=self.N, sample_size=1, isInt=True, sample_space=(maxXY, minXY))
        data = utils.minmax_norm(data)[0]

        self._fillEntry(data=data, entrylist=self.genDmViews)

        self.showingTrueLabel = True
        self._toggleTrueLabel()

    def _fillEntry(self, data, entrylist):

        data = numpy.array(data).reshape(-1)

        for i in range(len(data)):
            entrylist[i].delete(0, END)
            entrylist[i].insert(0, round(data[i], 4))

    def _getEntry(self, entrylist):

        try:
            data = numpy.array([float(e.get())for e in entrylist])
        except:
            data = []

        return data

    def _plotRemove(self, label=None):

        if label is None:
            label = self.opt_method.get()
            label = 'Deep MDS' if label.split('.')[-1] == 'model' else label

        if label in self.dot_list.keys():
            self.dot_list[label].remove()

            for i in range(self.N):
                self.anno_list[label][i].remove()

            self.dot_list.pop(label)
            self.anno_list.pop(label)

        if len(self.dot_list) > 0:
            self.ax.legend()
        elif self.ax.get_legend() is not None:
            self.ax.get_legend().remove()

        self.canvas.draw()

    def _plotClear(self):

        for label in self.methods.keys():
            self._plotRemove(label)

    def _plotData(self, x, y, label):

        anno = []

        self._plotRemove(label)

        self.dot_list[label] = self.ax.scatter(
            x, y, label=label, c=self.methods_color[label])

        for i in range(self.N):
            anno.append(self.ax.annotate(i, (x[i], y[i])))

        self.anno_list[label] = anno
        self.ax.legend()
        self.canvas.draw()

    def _colorize_diff(self, entrylist):

        try:
            diff = self._getEntry(self.rsDmViews) - \
                self._getEntry(self.genDmViews)
        except:
            return

        diff = numpy.abs(diff)

        color = 30 + int(150 - 30) * diff

        for i in range(len(color)):

            if self.colorized == True:
                entrylist[i].config(background="white")
            else:
                entrylist[i].config(
                    background="#9E{0:02X}{0:02X}".format(int(color[i])))

        self.colorized = not self.colorized

    def _toggleTrueLabel(self):

        if self.showingTrueLabel:
            self._plotRemove('True Data')
            self.btn_toggleTrueLabel.config(text='Show True Data')
            self.showingTrueLabel = False
        elif self.coords is not None:
            self._plotData(
                self.coords[0, :, 0],
                self.coords[0, :, 1], 'True Data')
            self.btn_toggleTrueLabel.config(text='Hide True Data')
            self.showingTrueLabel = True


root = Tk()
root.minsize(width=1150, height=600)
root.resizable(0, 0)

root.configure(background="white")

maxXY, minXY = (1000, 1)

app = Application(N=10, d=2, master=root)

root.mainloop()


def shutdown():
    app.destroy()


app.protocol("WM_DELETE_WINDOW", shutdown)
