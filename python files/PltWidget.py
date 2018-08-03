# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:27:01 2018

@author: Doug
"""
import pyqtgraph as pg
from CustomViewBox import CustomViewBox

class PltWidget(pg.GraphicsLayout):
    """
    Subclass of PlotWidget
    """
    def __init__(self, parent=None):
        """
        Constructor of the widget
        """
        super(PltWidget, self).__init__()
        self.addViewBox(CustomViewBox())