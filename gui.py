# %%
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

color = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Application:

    def __init__(self, N, d, master=None):

        self.N = N
        self.d = d

        self.master = master

        self.etm = manifold.Algorithm(N, d)
        # self.etm.use_pretrained_model('result\Coord_StepLinear_E2_3MSE_5_82_400.model')

        self.methods = {'Classical MDS':    self.etm.classicalMDS,
                        'Isomap':           self.etm.isomap,
                        'Landmark MDS':     self.etm.landmarkMDS,
                        'Non-Metric MDS':   self.etm.nonMetricMDS,
                        'Fastmap':          self.etm.fastmap,
                        'Deep MDS':         self.etm.deepMDS
                        }

        self.methods_name = [*self.methods.keys()]
        self.methods_color = dict(
            zip(self.methods_name, color[:len(self.methods_name)]))
        self.methods_color['True Data'] = 'b'

        self.opt_loss = StringVar(self.master)
        self.opt_model = StringVar(self.master)
        self.opt_prep = StringVar(self.master)
        self.opt_method = StringVar(self.master)

        self.opt_loss.set(list(manifold.lossFunMap.keys())[0])
        self.opt_model.set(list(manifold.modelMap.keys())[0])
        self.opt_prep.set(list(manifold.prepMap.keys())[0])
        self.opt_method.set('Select a method :')

        self.layerValue = IntVar()
        self.neuronValue = IntVar()
        self.minEpochValue = IntVar()
        self.maxEpochValue = IntVar()

        self.window_deepMDS = None

        self.dot_list = {}
        self.anno_list = {}

        self.colorized = False
        self.showingTrueLabel = False

        self.__buildDmPane().pack(side=LEFT, expand=None, fill=BOTH, padx=(20, 0))
        self.__buildCoordPane().pack(side=TOP, expand=Y, fill=BOTH,
                                    padx=(10, 20), pady=(30, 0))
        self.__buildPlotPane().pack(side=BOTTOM, expand=YES, fill=BOTH)

        self.__btn_genDM_fun()

    def __buildPlotPane(self):

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
            plotPane, text='Clear Current Label', command=self.plotRemove)
        btn_removeLabel.grid(
            row=4, column=1, columnspan=1, pady=10, padx=(30, 0), sticky='nesw')

        btn_removeLabel = Button(
            plotPane, text='Clear All Label', command=self.plotClear)
        btn_removeLabel.grid(
            row=4, column=2, columnspan=1, pady=10, padx=(10, 10), sticky='nesw')

        self.btntoggleTrueLabel = Button(plotPane, text='Hide True Label',
                                          command=self.toggleTrueLabel)
        self.btntoggleTrueLabel .grid(
            row=4, column=3, columnspan=1, pady=10, padx=(0, 30), sticky='nesw')

        return plotPane

    def __buildCoordPane(self):

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

        meun_method = OptionMenu(
            coordPane, self.opt_method, *self.methods_name, command=self.__btn_rsCoord_fun)
        meun_method.grid(
            row=gridy, column=1, columnspan=gridy, pady=10, padx=10, sticky='ew')
        meun_method.config(width=20)

        return coordPane

    def __buildDmPane(self):

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

        __btn_genEuDM = Button(
            dmPane_gen, text='Generate Euclidean Data', command=self.__btn_genEuDM)
        __btn_genEuDM.grid(
            row=gridy, column=0, columnspan=gridy + 1, pady=(10, 0), padx=30, sticky='nesw')

        btn_genDM = Button(
            dmPane_gen, text='Generate Random Data', command=self.__btn_genDM_fun)
        btn_genDM.grid(
            row=gridy + 1, column=0, columnspan=gridy + 1, pady=(0, 10), padx=30, sticky='nesw')

        self.label_loss = Label(dmPane_rs, background='#b8b8b8')
        self.label_loss.grid(
            row=0, column=0, columnspan=gridy - 3, pady=10, padx=(30, 10), sticky='nesw')

        btn_colorize = Button(dmPane_rs, text='Colorize Distance',
                              command=lambda: self.toggleColorizeDiff(self.rsDmViews))
        btn_colorize.grid(
            row=0, column=gridy - 3, columnspan=3, pady=10, padx=(0, 30), sticky='nesw')

        return dmPane

    def __disablePrepForAE(self, model):

        if model == 'AE':
            self.menu_prep['menu'].entryconfigure('M', state="disabled")
            self.menu_prep['menu'].entryconfigure('E2', state="disabled")
        else:
            self.menu_prep['menu'].entryconfigure('M', state= "normal")
            self.menu_prep['menu'].entryconfigure('E2', state="normal")


    def __buildDeepMDSWindow(self):

        self.window_deepMDS = Toplevel(self.master)
        self.window_deepMDS.resizable(0, 0)

        self.window_deepMDS.configure(background="white")
        self.window_deepMDS.grid_columnconfigure((0, 7), weight=1)

        menu_loss = OptionMenu(self.window_deepMDS,
                               self.opt_loss, *manifold.lossFunMap)

        menu_model = OptionMenu(self.window_deepMDS,
                                self.opt_model, *manifold.modelMap, command=self.__disablePrepForAE)

        self.menu_prep = OptionMenu(self.window_deepMDS,
                               self.opt_prep, *manifold.prepMap)

        label_loss = Label(self.window_deepMDS, text="Select a Loss function",
                           justify=LEFT, anchor="w",
                           width=17, background="white")
        label_model = Label(self.window_deepMDS, text="Select a Model",
                            justify=LEFT, anchor="w",
                            width=17, background="white")
        label_prep = Label(self.window_deepMDS, text="Select a Preprocess",
                           justify=LEFT, anchor="w",
                           width=17, background="white")

        self.entry_layer = Scale(self.window_deepMDS, background="white",
                                 from_=2, to=10, showvalue=0, variable=self.layerValue,
                                 label='Select the number of extra layer:', orient=HORIZONTAL)
        label_layer = Entry(self.window_deepMDS, width=5,
                            textvariable=self.layerValue)

        self.entry_nuerons = Scale(self.window_deepMDS, background="white",
                                   from_=-5, to=200, showvalue=0, variable=self.neuronValue,
                                   label='Select the number of extra neruons:', orient=HORIZONTAL)
        label_neuron = Entry(self.window_deepMDS, width=5,
                             textvariable=self.neuronValue)

        self.entry_minEpoch = Scale(self.window_deepMDS, background="white",
                                    resolution=10,
                                    from_=10, to=2000, showvalue=0, variable=self.minEpochValue,
                                    label='Select the number of min epoch:', orient=HORIZONTAL)
        label_minEpoch = Entry(self.window_deepMDS, width=5,
                               textvariable=self.minEpochValue)

        self.entry_maxEpoch = Scale(self.window_deepMDS, background="white",
                                    resolution=10,
                                    from_=10, to=2000, showvalue=0, variable=self.maxEpochValue,
                                    label='Select the number of max epoch:', orient=HORIZONTAL)
        label_maxEpoch = Entry(self.window_deepMDS, width=5,
                               textvariable=self.maxEpochValue)

        btn_pretrained = Button(
            self.window_deepMDS, text='Using Pre-trained Model', command=self.__promptForModel)

        btn_submit = Button(
            self.window_deepMDS, text='Apply and Close', command=self.__configDeepModel)

        label_model.grid(row=0, column=1, columnspan=1,
                         pady=10, padx=10, sticky='ew')
        menu_model .grid(row=0, column=2, columnspan=1,
                         pady=10, padx=10, sticky='ew')

        label_prep.grid(row=1, column=1, columnspan=1,
                        pady=10, padx=10, sticky='ew')
        self.menu_prep.grid(row=1, column=2, columnspan=1,
                       pady=10, padx=10, sticky='ew')

        label_loss.grid(row=2, column=1, columnspan=1,
                        pady=10, padx=10, sticky='ew')
        menu_loss.grid(row=2, column=2, columnspan=1,
                       pady=10, padx=10, sticky='ew')

        self.entry_layer.grid(row=3, column=1, columnspan=5,
                              pady=10, padx=10, sticky='ew')
        self.entry_nuerons.grid(
            row=4, column=1, columnspan=5, pady=10, padx=10, sticky='ew')
        self.entry_minEpoch.grid(
            row=5, column=1, columnspan=5, pady=10, padx=10, sticky='ew')
        self.entry_maxEpoch.grid(
            row=6, column=1, columnspan=5, pady=10, padx=10, sticky='ew')

        label_layer.grid(row=3, column=6, columnspan=1,
                         pady=10, padx=10, sticky='ew')
        label_neuron.grid(row=4, column=6, columnspan=1,
                          pady=10, padx=10, sticky='ew')
        label_minEpoch.grid(row=5, column=6, columnspan=1,
                            pady=10, padx=10, sticky='ew')
        label_maxEpoch.grid(row=6, column=6, columnspan=1,
                            pady=10, padx=10, sticky='ew')

        btn_pretrained.grid(row=7, column=1, columnspan=3,
                        pady=10, padx=10, sticky='w')

        btn_submit.grid(row=7, column=4, columnspan=3,
                        pady=10, padx=10, sticky='w')

    def __btn_rsCoord_fun(self, method_name=None):

        dm = self.getEntry(self.genDmViews).reshape(self.N, self.N)

        method_name = self.opt_method.get() if method_name is None else method_name

        if method_name == 'Deep MDS':

            if self.window_deepMDS is not None:
                return 

            self.plotRemove(method_name)
            self.__buildDeepMDSWindow()

            self.master.wait_window(self.window_deepMDS)
            self.window_deepMDS = None

        data, time = utils.time_measure(self.methods[method_name], [dm])

        data = torch.tensor(data)
        data = utils.minmax_norm(data.view(1, -1))[0]

        self.fillEntry(data=data, entrylist=self.rsCoordViews)

        coords = data.reshape(self.N, self.d)
        self.plotData(x=coords[:, 0], y=coords[:, 1], label=method_name)

        dm = torch.tensor(dm)

        rs_dm = utils.get_distanceSq_matrix(coords) ** 0.5
        rs_dm = utils.minmax_norm(rs_dm)[0] * torch.max(dm)

        loss = torch.mean(torch.abs(dm - rs_dm.view_as(dm)))

        self.label_loss.config(text='Loss = %s (%ss)' % (
            str(round(float(loss), 10)), str(round(time, 4))
        ))

        self.fillEntry(data=rs_dm, entrylist=self.rsDmViews)

        self.colorized = not self.colorized
        self.toggleColorizeDiff(self.rsDmViews)

    def __btn_genEuDM(self):

        self.plotClear()

        self.coords = dataSource.get_rand_data((1, self.N, 2), isInt=True,
                                               maxXY=maxXY, minXY=minXY)
        self.coords = utils.minmax_norm(self.coords)[0]

        data = utils.get_distanceSq_matrix(self.coords) ** 0.5
        data = utils.minmax_norm(data)[0]

        self.fillEntry(data=data, entrylist=self.genDmViews)

        self.coords = self.etm.norm(self.coords[0].detach().numpy())

        self.showingTrueLabel = False
        self.toggleTrueLabel()

        self.resetResult()

    def __btn_genDM_fun(self):

        self.plotClear()

        self.coords = None

        data = dataSource.generate_rand_DM(
            N=self.N, sample_size=1, isInt=True, sample_space=(maxXY, minXY))
        data = utils.minmax_norm(data)[0]

        self.fillEntry(data=data, entrylist=self.genDmViews)

        self.showingTrueLabel = True
        self.toggleTrueLabel()

        self.resetResult()

    def __configDeepModel(self):

        self.etm.make_new_model(
            modelKey=self.opt_model.get(), lossFunKey=self.opt_loss.get(), prepKey=self.opt_prep.get(),
            nNeuron=int(self.entry_layer.get()), nLayer=int(self.entry_nuerons.get()),
            minEpoch=int(self.entry_minEpoch.get()), maxEpoch=int(self.entry_maxEpoch.get())
        )

        self.window_deepMDS.destroy()

    def __promptForModel(self):

        model_path = filedialog.askopenfilename(
            initialdir=".", title="Select A Model", filetypes=(("Pretained Model", "*.model"), ("all files", "*.*")))

        if model_path != '':
            self.opt_method.set(model_path.split('\\')[-1].split('/')[-1])
            self.etm.use_pretrained_model(model_path)
            self.window_deepMDS.destroy()
    
    def fillEntry(self, data, entrylist):

        data = numpy.array(data).reshape(-1)
        datalen = len(data)

        for i in range(len(entrylist)):

            entrylist[i].delete(0, END)

            if i >= datalen:
                entrylist[i].insert(0, '')
            else:
                entrylist[i].insert(0, round(data[i], 4))

    def getEntry(self, entrylist):

        try:
            data = numpy.array([float(e.get())for e in entrylist])
        except:
            data = []

        return data

    def plotRemove(self, label=None):

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

    def plotClear(self):

        for label in self.methods.keys():
            self.plotRemove(label)

    def plotData(self, x, y, label):

        anno = []

        self.plotRemove(label)

        self.dot_list[label] = self.ax.scatter(
            x, y, label=label, c=self.methods_color[label])

        for i in range(self.N):
            anno.append(self.ax.annotate(i, (x[i], y[i])))

        self.anno_list[label] = anno
        self.ax.legend()
        self.canvas.draw()

    def resetResult(self):

        self.colorized = True
        self.toggleColorizeDiff(self.rsDmViews)

        self.label_loss.config(text='')

        self.fillEntry(data=[], entrylist=self.rsDmViews)
        self.fillEntry(data=[], entrylist=self.rsCoordViews)

    def toggleColorizeDiff(self, entrylist):

        try:
            diff = self.getEntry(self.rsDmViews) - \
                self.getEntry(self.genDmViews)
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

    def toggleTrueLabel(self):

        if self.showingTrueLabel:
            self.plotRemove('True Data')
            self.btntoggleTrueLabel.config(text='Show True Data')
            self.showingTrueLabel = False
        elif self.coords is not None:
            self.plotData(
                self.coords[:, 0],
                self.coords[:, 1], 'True Data')
            self.btntoggleTrueLabel.config(text='Hide True Data')
            self.showingTrueLabel = True


root = Tk()
root.minsize(width=1150, height=600)
root.resizable(0, 0)

root.configure(background="white")

maxXY, minXY = (1000, 1)

app = Application(N=10, d=2, master=root)

root.mainloop()


app.protocol("WM_DELETE_WINDOW", lambda x: x.destroy())


# %%
