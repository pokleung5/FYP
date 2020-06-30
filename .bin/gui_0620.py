import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

import numpy as np

import utils

matplotlib.use('TkAgg')

class Application(tk.Frame):
    def __init__(self, N, master=None):
        tk.Frame.__init__(self, master)
        
        self.N = N
        self.master = master

        # self.dmPane = tk.Canvas(self.master)
        # self.coordPane = tk.Canvas(self.master)
        self.createWidgets(self.master)

    def createWidgets(self, master):
        fig=plt.figure(figsize=(8,8))
        ax=fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
        
        canvas=FigureCanvasTkAgg(fig,master=master)
        canvas.get_tk_widget().grid(row=0,column=1)
        canvas.draw()

        self.plotbutton=tk.Button(master=master, text="plot", command=lambda: self.plot(canvas,ax))
        self.plotbutton.grid(row=0, column=0)

    def plotOrigCoord(self, coord: np.array):
        pass

    def plotRsCoord(self, coord: np.array):
        pass

    def plot(self,canvas,ax):
        c = ['r','b','g']  # plot marker colors
        ax.clear()         # clear axes from previous plot
        for i in range(3):
            theta = np.random.uniform(0,360,self.N)
            r = np.random.uniform(0,1,self.N)
            ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
            canvas.draw()

root = tk.Tk()
app=Application(N=10, master=root)
app.mainloop()