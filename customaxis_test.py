# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:33:02 2018

@author: Doug
"""

import PyQt5.QtCore as QtCore
import pyqtgraph as pg
import numpy as np

class MyStringAxis(pg.AxisItem):
    def __init__(self, xdict, *args, **kwargs):
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.x_values = np.asarray(xdict.keys())
        self.x_strings = xdict.values()

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            # vs is the original tick value
            vs = v * scale
            # if we have vs in our values, show the string
            # otherwise show nothing
            if vs in self.x_values:
                # Find the string with x_values closest to vs
                vstr = self.x_strings[np.abs(self.x_values-vs).argmin()]
            else:
                vstr = ""
            strings.append(vstr)
        return strings

x = ['a', 'b', 'c', 'd', 'e', 'f']
y = [1, 2, 3, 4, 5, 6]
xdict = dict(enumerate(x))

win = pg.GraphicsWindow()
stringaxis = MyStringAxis(xdict, orientation='bottom')
plot = win.addPlot(axisItems={'bottom': stringaxis})
curve = plot.plot(list(xdict.keys()),y)

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()