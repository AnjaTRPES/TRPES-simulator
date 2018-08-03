import sys
import math
from PyQt5.QtWidgets import QApplication, QDialog,QMainWindow,QMessageBox,QFileDialog
from PyQt5.QtCore import pyqtSlot
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
from ui_TRPESsimulator_main import Ui_TRPES_simulator

import pyqtgraph as pg
pg.setConfigOption('background','w')
pg.setConfigOption('foreground','k')
import numpy as np
from scipy.optimize import leastsq
from matplotlib import cm
from scipy import interpolate
import copy as copy

from skewed_gauss import skewed_gauss
from fit_class import fits
from fit_class_2D import fits_2D
from fit_class_2D_2nd import fits_2D_2nd
from fit_class_bidir import fits_bidir
from fit_class_bidir_2d import fits_bidir_2d

import json
from anja_json_decoder import encode_anja,decode_anja

'''
#defining how floating-points error will be handled
def err_handler(type,flag):
    print('called it,YOOOOOOOOOOOOOOOOOOOO')
    return 0
saved_handler=np.seterrcall(err_handler)
saved_err=np.seterr(over='call')
'''
np.seterr(over='ignore')

class TRPES_simulator(QMainWindow,Ui_TRPES_simulator):
    def __init__(self,parent=None):
        super(TRPES_simulator,self).__init__(parent)
        #self.ui=Ui_TRPES_simulator()
        self.setupUi(self)
        QApplication.restoreOverrideCursor()
        self.initalization_finished=False
        self.dir='home'
        self.switch_TRPES_axis=False
        self.inverse_loading=False
        self.display_log_t=False
        #all variables for simulate TRPES here
        #all variables pe
        self.pe_gauss=[[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)]]
        self.pe_energy_range=np.arange(0.,5.,(5.-0.)/250)
        self.pe_spectra=[self.pe_gauss[0][0].calculate_gauss(self.pe_energy_range),
                         self.pe_gauss[0][0].calculate_gauss(self.pe_energy_range),
                         self.pe_gauss[0][0].calculate_gauss(self.pe_energy_range),
                         self.pe_gauss[0][0].calculate_gauss(self.pe_energy_range)]
        self.colors=[(255,0,0),(76,153,0),(0,255,255),(0,0,204),
                     (153,0,153),(255,0,127),(255,255,51),(255,128,0),
                     (255,0,0),(76,153,0),(0,255,255),(0,0,204),
                     (153,0,153),(255,0,127),(255,255,51),(255,128,0)]
        self.initialize_pe_graph()
        #all variables time
        self.viewer_time_legend=self.viewer_time.addLegend(size=(1.,0.1))
        self.temporal_offset=0.
        self.simulated_fit=fits()
        self.update_fit_parameters()
        self.time_fit_range=[]
        self.update_time_fit_range()
        self.time_curves=[]
        self.initialize_time_graph()
        #variables 2D TRPES
        self.sim_2D_TRPES=None
        self.initialize_sim_2D_TRPES()
        ##all variables for simulating TRPES in both directions
        self.bidir_legends_pe=[self.viewer_pe_bidir_pos.addLegend(size=(1.,0.1),offset=(-0.4,0.4)),self.viewer_pe_bidir_min.addLegend(size=(1.,0.1),offset=(-0.4,0.4))]
        self.bidir_legends_time=[self.viewer_time_bidir_pos.addLegend(size=(1.,0.1),offset=(-0.4,0.4)),self.viewer_time_bidir_min.addLegend(size=(1.,0.1),offset=(-0.4,0.4))]
        self.pe_gauss_bidir=[[[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)]],
                             [[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)]]]
        stepsize=(-self.doubleSpinBox_time_min_bidir_pos.value()+self.doubleSpinBox_time_max_bidir_pos.value())/200.
        self.time_fit_range_bidir=np.arange(self.doubleSpinBox_time_min_bidir_pos.value(),self.doubleSpinBox_time_max_bidir_pos.value(),stepsize)       
        self.pe_energy_range_bidir=np.arange(0.,5.,(5.-0.)/250.)
        self.default_pe_spectra=skewed_gauss(0.0,0.2,0.,1.).calculate_gauss(self.pe_energy_range_bidir)
        self.pe_spectra_bidir=[[self.default_pe_spectra,self.default_pe_spectra,self.default_pe_spectra,self.default_pe_spectra],
                               [self.default_pe_spectra,self.default_pe_spectra,self.default_pe_spectra,self.default_pe_spectra]]
        self.temporal_offset_bidir=0.
        self.simulated_fit_bidir=[fits(),fits()]
        self.time_curves_bidir=[[],[]]
        self.sim_2D_TRPES_bidir=None
        
        self.update_fit_params_bidir()
        self.update_time_fit_range_bidir()
     
        self.initialize_bidir_graphs()
        #all variables SVD analysis
        self.SVD_original_for_bg=None
        self.SVD_TRPES=None
        self.SVD_time=None
        self.SVD_eV=None
        self.SVD_time_rec=None
        self.SVD_eV_rec=None
        self.SVD_values=None
        self.SVD_reconstructed=None
        self.initialize_SVD_graphs()
        #all variables global analysis
        self.global_TRPES=None
        self.global_time=None
        self.global_eV=None
        self.global_fit=fits()
        self.initialize_global_graphs()
        self.global_pe_gauss=[[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)],[skewed_gauss(0.0,0.2,0.,1.)]]
        self.global_TRPES_fitted=None
        self.global_TRPES_residual=None
        self.global_params_from_fit=None
        self.global_DAS_fitted=None
        self.global_decays_fitted=None
        self.test_fit=None
        self.global_t0=None
        self.global_t_for_fit=None
        self.global_eV_for_fit=None
        self.gobal_guess_time_curves=None
        self.global_TRPES_original_for_bg=None
        self.initalization_finished=True
        self.global_TRPES_dataset=None
        self.global_time_dataset=None
        self.global_eV_dataset=None
        #all variables global analysis bidir
        self.global_TRPES_bidir=None
        self.global_time_bidir=None
        self.global_eV_bidir=None
        self.global_TRPES_original_for_bg_bidir=None
        self.initalize_global_graphs_bidir()
        self.global_TRPES_fitted_bidir=None
        self.global_fit_bidir=fits_bidir()
        self.global_fit_bidir_2d=fits_bidir_2d()
        self.global_total_pe_decay_bidir=None
        self.global_fit_time_offset=[]
        self.global_DAS_bidir=[[],[]]
        self.global_decays_bidir=[[],[]]
        self.global_time_bidir_fitted=None
        self.global_TRPES_bidir_fitted=None
        self.global_TRPES_residual_bidir=None
        self.global_TRPES_bidir_dataset=None
        self.global_time_bidir_dataset=None
        self.global_eV_bidir_dataset=None
        

    @pyqtSlot()
    def on_info_triggered(self):
        print('called?')
        text_to_display='This TRPES simulator with global fitting routines was\n'
        text_to_display+='designed and programmed by Anja Roeder. For any suggestions,\n'
        text_to_display+='problems, desired functionalities and especially errors please\n'
        text_to_display+='send a mail to aroder@uottawa.ca.\nI will try to respond as soon as possible!\nCheers, Anja'
        QMessageBox.about(self,'info',text_to_display)
        
    @pyqtSlot('bool')
    def on_actionSwitch_Csv_file_orientation_toggled(self,value):
        self.inverse_loading=value
    ##########################
    #all functions regrouping from simulate_trpes
    ##########################

    #----------------------------------
    #all functions regrouped from the tab PE
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_max_Energy_PE_valueChanged(self,value):
        #update the energy range
        self.pe_energy_range=np.arange(0.,value,(value)/250.)
        self.update_pe_graph()   
        
    @pyqtSlot('int')
    def on_comboBox_DisplayedComponent_currentIndexChanged(self,index):
        print('do stuff')
        self.spinBox_PE_number.blockSignals(True)
        self.comboBox_PE_current.blockSignals(True)
        self.spinBox_PE_number.setValue(len(self.pe_gauss[index]))
        self.comboBox_PE_current.clear()
        print('jo')
        for i,n in enumerate(self.pe_gauss[index]):
            self.comboBox_PE_current.addItem(str(i+1))
        print('hm')
        self.comboBox_PE_current.setCurrentIndex(self.comboBox_PE_current.count()-1)
        self.spinBox_PE_number.blockSignals(False)
        self.comboBox_PE_current.blockSignals(False)
        self.load_pe_gauss()
        self.update_pe_graph()
        
    @pyqtSlot('int')
    def on_comboBox_global_pick_model_2_currentIndexChanged(self,value):
        print('do other stuff')
        self.update_time_graph()
        self.update_pe_graph() 

        

    @pyqtSlot('int')
    def on_spinBox_PE_number_valueChanged(self,value):
        #add a new one to the self.pe_gauss or delete it
        self.comboBox_PE_current.blockSignals(True)
        if value>len(self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()]):
            self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()].append(skewed_gauss(0.,0.2,0.,1.))            
            #and update self.comboBox_PE_current
            self.comboBox_PE_current.addItem(str(value))
        else:
            print('removing stuff')
            del self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][-1]
            #remove from combobox
            self.comboBox_PE_current.removeItem(self.comboBox_PE_current.count()-1)
        #put combobox and last index
        self.comboBox_PE_current.setCurrentIndex(self.comboBox_PE_current.count()-1)
        self.comboBox_PE_current.blockSignals(False)
        self.load_pe_gauss()
        #update graphs etc
        self.update_pe_gauss()
        self.update_pe_graph()

        
    @pyqtSlot('int')
    def on_comboBox_PE_current_currentIndexChanged(self,index):                              
        self.load_pe_gauss()
        self.update_pe_graph()
        

    @pyqtSlot('double')
    def on_doubleSpinBox_PE_center_valueChanged(self,value):
        self.update_pe_gauss()
        self.update_pe_graph()

    @pyqtSlot('double')
    def on_doubleSpinBox_PE_width_valueChanged(self,value):
        self.update_pe_gauss()
        self.update_pe_graph()

    @pyqtSlot('double')
    def on_doubleSpinBox_pe_assymetry_valueChanged(self,value):
        self.update_pe_gauss()
        self.update_pe_graph()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_pe_rel_intensity_valueChanged(self,value):
        self.update_pe_gauss()
        self.update_pe_graph()

    def update_pe_gauss(self):
        #update with new values the gauss list
        index=self.comboBox_PE_current.currentIndex()
        self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index].mu=self.doubleSpinBox_PE_center.value()
        self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index].fwhm=self.doubleSpinBox_PE_width.value()
        self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index].alpha=self.doubleSpinBox_pe_assymetry.value()
        self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index].intensity=self.doubleSpinBox_pe_rel_intensity.value()
        print('intensity',self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index].intensity)

    def load_pe_gauss(self):
        #display the current gauss(or neutral)
        self.doubleSpinBox_PE_center.blockSignals(True)
        self.doubleSpinBox_PE_width.blockSignals(True)
        self.doubleSpinBox_pe_assymetry.blockSignals(True)
        self.doubleSpinBox_pe_rel_intensity.blockSignals(True)
        index=self.comboBox_PE_current.currentIndex()
        current_gauss=self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()][index]
        print('values to set:',current_gauss.mu, current_gauss.fwhm, current_gauss.alpha,current_gauss.intensity)
        self.doubleSpinBox_PE_center.setValue(current_gauss.mu)
        self.doubleSpinBox_PE_width.setValue(current_gauss.fwhm)
        self.doubleSpinBox_pe_assymetry.setValue(current_gauss.alpha)
        self.doubleSpinBox_pe_rel_intensity.setValue(current_gauss.intensity)
        self.doubleSpinBox_PE_center.blockSignals(False)
        self.doubleSpinBox_PE_width.blockSignals(False)
        self.doubleSpinBox_pe_assymetry.blockSignals(False)
        self.doubleSpinBox_pe_rel_intensity.blockSignals(False)
    
    def update_pe_graph(self):
        #update the pe graph with new values
        #first make the sum
        summed_pe=np.zeros(self.pe_energy_range.shape)
        for gauss in self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()]:
            summed_pe+=gauss.calculate_gauss(self.pe_energy_range)
        normalize_factor=max(summed_pe)
        summed_pe=summed_pe/normalize_factor
        """
        #delete the old legend
        try:
            self.viewer_pe_legend.scene().removeItem(self.viewer_pe_legend)
        except Exception as e:
           pass
        """
        self.viewer_pe_legend=self.viewer_pe.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        #plot the summed_pe
        self.viewer_pe.clear()
        for i, gauss in enumerate(self.pe_gauss[self.comboBox_DisplayedComponent.currentIndex()]):
            if i==(self.comboBox_PE_current.currentIndex()):
                c=self.viewer_pe.plot(self.pe_energy_range,0.8*gauss.calculate_gauss(self.pe_energy_range),
                                pen=pg.mkPen(self.colors[i],width=3),name='Gauss'+str(i))
            else:
                c=self.viewer_pe.plot(self.pe_energy_range,0.8*gauss.calculate_gauss(self.pe_energy_range),
                                pen=self.colors[i],width=1,name='Gauss'+str(i)) 
        self.viewer_pe.plot(self.pe_energy_range,summed_pe,pen=(0,0,0))
        #update the definitions stuff
        self.pe_spectra[self.comboBox_DisplayedComponent.currentIndex()]=summed_pe
        if self.initalization_finished==True:
            self.update_sim_2D_TRPES()

    def initialize_pe_graph(self):
        self.viewer_pe.setLabel('left', "intensity", units='arb.u.')
        self.viewer_pe.setLabel('bottom', "energy", units='eV')
        self.viewer_pe.plotItem.ctrlMenu=[]
        self.update_pe_graph()
    
    #----------------------------------
    #all functions regrouped from the tab time
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau1_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()    
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau2_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()    
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau3_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()        
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_1_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()        
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_2_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_3_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_irf_valueChanged(self,value):
        self.update_fit_parameters()
        self.update_time_graph()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_pos_irf_valueChanged(self,value):
        self.temporal_offset=value
        self.update_fit_parameters()
        self.update_time_graph()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_offset_valueChanged(self,value):        
        self.update_fit_parameters()
        self.update_time_graph()
        
    
    @pyqtSlot('bool')
    def on_radioButton_no_offset_toggled(self,value):
        self.offset_change()
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_toggled(self,value):
        self.offset_change()
    @pyqtSlot('bool')
    def on_radioButton_time_offset_toggled(self,value):
        self.offset_change()

    @pyqtSlot('int')
    def on_radiobutton_group_time_offset_buttonClicked(self,value):
        self.offset_change()
    
    def offset_change(self):
        self.update_time_graph()

    
    def update_fit_parameters(self):
        """"
        getting the currently displayed fit parameters and updating the self.simulated_fit object
        """
        self.simulated_fit.tau1=self.doubleSpinBox_time_tau1.value()
        self.simulated_fit.tau2=self.doubleSpinBox_time_tau2.value()
        self.simulated_fit.tau3=self.doubleSpinBox_time_tau3.value()
        self.simulated_fit.sigma_1=[self.doubleSpinBox_time_int_1.value()]
        self.simulated_fit.sigma_2=[self.doubleSpinBox_time_int_2.value()]
        self.simulated_fit.sigma_3=[self.doubleSpinBox_time_int_3.value()]
        self.simulated_fit.fwhm=[self.doubleSpinBox_time_irf.value()]
        self.simulated_fit.moy=[0.]
        self.simulated_fit.time_offset=[self.doubleSpinBox_time_pos_irf.value()]
        self.simulated_fit.sigma_offset=[self.doubleSpinBox_time_int_offset.value()]
    
    def update_time_graph(self):
        """
        updating the time plot
        """
        #calculating the necessary curves
        self.time_curves=[]
        t=self.time_fit_range+self.temporal_offset
        all_curves_taus=[self.simulated_fit.mono_exp_decay(t,0),
                    self.simulated_fit.bi_exp_decay_population(t,0),
                    self.simulated_fit.tri_exp_decay_population(t,0)]
        names_all=['mono_exp','bi-exp','tri-exp']
        all_curves_final_offset=[self.simulated_fit.mono_exp_decay_final_state_pop(t,0),
                                 self.simulated_fit.bi_exp_decay_final_state_pop(t,0),
                                 self.simulated_fit.tri_exp_decay_final_state_pop(t,0)]
        all_curves_offset=self.simulated_fit.offset(t,0)
        self.time_curves=all_curves_taus[:self.comboBox_global_pick_model_2.currentIndex()+1]
        names=names_all[:self.comboBox_global_pick_model_2.currentIndex()+1]
        if self.radioButton_time_final_offset.isChecked()==True:
            self.time_curves.append(all_curves_final_offset[self.comboBox_global_pick_model_2.currentIndex()])
            names.append('offset')
        elif self.radioButton_time_offset.isChecked()==True:
            self.time_curves.append(all_curves_offset)
            names.append('offset')
        #delete the old legend
        """
        try:
            self.viewer_time_legend.scene().removeItem(self.viewer_time_legend)
        except Exception as e:
           print(e)
           pass
        """
        self.viewer_time_legend=self.viewer_time.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        #make a sum curve
        summed_decay=np.zeros(self.time_curves[0].shape)
        for curve in self.time_curves:
            summed_decay+=curve
        #plot that stuff
        self.viewer_time.clear()
        for i,curve in enumerate(self.time_curves):
            self.viewer_time.plot(self.time_fit_range,0.8*curve,name=names[i],
                                pen=pg.mkPen(self.colors[i],width=2)) 
        self.viewer_time.plot(self.time_fit_range,summed_decay,pen=pg.mkPen((0,0,0),width=2),name='sum')
        if self.initalization_finished==True:
            self.update_sim_2D_TRPES()

    
    def initialize_time_graph(self):
        self.viewer_time.setLabel('left', "intensity", units='arb.u.')
        self.viewer_time.setLabel('bottom', "time", units='ps')
        self.viewer_time.plotItem.ctrlMenu=[]
        self.update_time_graph()
    
    @pyqtSlot('double')
    def on_doubleSpinBox_time_max_valueChanged(self,value):
        self.doubleSpinBox_time_min.blockSignals(True)
        self.doubleSpinBox_time_max.blockSignals(True)
        if value<self.doubleSpinBox_time_min.value():
            self.doubleSpinBox_time_max.setValue(self.doubleSpinBox_time_min.value()+1)
        self.doubleSpinBox_time_min.blockSignals(False)
        self.doubleSpinBox_time_max.blockSignals(False)
        self.update_time_fit_range()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_min_valueChanged(self,value):
        self.doubleSpinBox_time_min.blockSignals(True)
        self.doubleSpinBox_time_max.blockSignals(True)
        if value>self.doubleSpinBox_time_max.value():
            self.doubleSpinBox_time_min.setValue(self.doubleSpinBox_time_max.value()-1)
        self.doubleSpinBox_time_min.blockSignals(False)
        self.doubleSpinBox_time_max.blockSignals(False)
        self.update_time_fit_range()
    
    def update_time_fit_range(self):
        """
        updating the time_fit_range, make it so that there will be about 200 points
        """
        stepsize=(self.doubleSpinBox_time_max.value()-self.doubleSpinBox_time_min.value())/200
        self.time_fit_range=np.arange(self.doubleSpinBox_time_min.value(),self.doubleSpinBox_time_max.value(),stepsize)
        self.update_time_graph()
    
    def initialize_sim_2D_TRPES(self):
        self.update_sim_2D_TRPES()
        
    def update_sim_2D_TRPES(self):
        #calculate the 2D TRPES from the considered components, pe &time
        ones_matrix=np.ones((self.pe_energy_range.shape[0],self.time_fit_range.shape[0]))
        #print('pe',self.pe_energy_range.shape,'time',self.time_fit_range.shape)
        self.sim_2D_TRPES=np.zeros((self.pe_energy_range.shape[0],self.time_fit_range.shape[0]))
        #sum all the individuals up to make the final thing
        for i in range(self.comboBox_global_pick_model_2.currentIndex()+1):
            intermed=np.reshape(self.pe_spectra[i],(self.pe_energy_range.shape[0],1))*ones_matrix*self.time_curves[i]
            self.sim_2D_TRPES+=intermed
  
        if self.radioButton_time_final_offset.isChecked()==True or self.radioButton_time_offset.isChecked()==True:
            intermed=np.reshape(self.pe_spectra[-1],(self.pe_energy_range.shape[0],1))*ones_matrix*self.time_curves[-1]
            self.sim_2D_TRPES+=intermed
       
        self.sim_2D_TRPES=self.sim_2D_TRPES.transpose()
        #add the noise if the user desired it
        if self.checkBox_random_noise.isChecked()==True:
            noise=noise = np.random.normal(0, self.doubleSpinBox_noise.value(), self.sim_2D_TRPES.shape)
            self.sim_2D_TRPES+=noise
        self.plot_TRPES(self.viewer_sim_trpes,self.sim_2D_TRPES,self.time_fit_range,self.pe_energy_range)


    @pyqtSlot('bool')
    def on_checkBox_random_noise_toggled(self,value):
        self.update_sim_2D_TRPES()

    @pyqtSlot('double')
    def on_doubleSpinBox_noise_valueChanged(self,value):
        self.update_sim_2D_TRPES()
    ##########################
    #all functions regrouping from simulate_trpes_both directions
    ##########################
    def initialize_bidir_graphs(self):
        for pe_viewer in [self.viewer_pe_bidir_pos,self.viewer_pe_bidir_min]:
            pe_viewer.setLabel('left', "intensity", units='arb.u.')
            pe_viewer.setLabel('bottom', "energy", units='eV')
            pe_viewer.plotItem.ctrlMenu=[]
        for time_viewer in [self.viewer_time_bidir_pos,self.viewer_time_bidir_min]:
            time_viewer.setLabel('bottom', "time", units='ps')
            time_viewer.setLabel('left', "intensity", units='arb.u.')
            time_viewer.plotItem.ctrlMenu=[]
        self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)
        self.update_time_graph_bidir(self.viewer_time_bidir_min,1)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_pos,0)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_min,1)
        self.update_sim_2D_TRPES_bidir()
   
    #----------------------------------
    #all functions regrouped from the tab PE pos
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_max_Energy_PE_bidir_pos_valueChanged(self,value):
        self.update_pe_energy_range_bidir(value,'pos')
        
    def update_pe_energy_range_bidir(self,value,which='pos'):
        self.doubleSpinBox_max_Energy_PE_bidir_min.blockSignals(True)
        self.doubleSpinBox_max_Energy_PE_bidir_pos.blockSignals(True)
        if which=='pos':
            self.doubleSpinBox_max_Energy_PE_bidir_min.setValue(value)
            self.pe_energy_range_bidir=np.arange(0.,value,value/250.)
        elif which=='min':
            self.doubleSpinBox_max_Energy_PE_bidir_pos.setValue(value)
            self.pe_energy_range_bidir=np.arange(0.,value,value/250.)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_pos,0)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_min,1)
        self.doubleSpinBox_max_Energy_PE_bidir_min.blockSignals(False)
        self.doubleSpinBox_max_Energy_PE_bidir_pos.blockSignals(False)

    def update_pe_viewer_bidir(self, viewer,n):
        '''
        showing the photoelectron spectra
        '''
        #update the pe graph with new values
        #first make the sum
        summed_pe=np.zeros(self.pe_energy_range_bidir.shape)
        current_components=[self.comboBox__DisplayedComponent_bidir_pos.currentIndex(),self.comboBox__DisplayedComponent_bidir_min.currentIndex()][n]
        for gauss in self.pe_gauss_bidir[n][current_components]:
            summed_pe+=gauss.calculate_gauss(self.pe_energy_range_bidir)
        normalize_factor=max(summed_pe)
        if normalize_factor==0.:
            normalize_factor=1.
        summed_pe=summed_pe/normalize_factor
        self.bidir_legends_pe[n]=viewer.addLegend(size=(1.,0.1),offset=(-0.4,0.4))

        #plot the summed_pe
        viewer.clear()
        current_index=[self.comboBox_PE_current_bidir_pos.currentIndex(),self.comboBox_PE_current_bidir_min.currentIndex()][n]
        for i, gauss in enumerate(self.pe_gauss_bidir[n][current_components]):
            if i==(current_index):
                viewer.plot(self.pe_energy_range_bidir,0.8*gauss.calculate_gauss(self.pe_energy_range_bidir),
                                pen=pg.mkPen(self.colors[i],width=3),name='Gauss'+str(i))
            else:
                viewer.plot(self.pe_energy_range_bidir,0.8*gauss.calculate_gauss(self.pe_energy_range_bidir),
                                pen=self.colors[i],width=1,name='Gauss'+str(i)) 
        viewer.plot(self.pe_energy_range_bidir,summed_pe,pen=(0,0,0))
        #update the definitions stuff
        self.pe_spectra_bidir[n][current_components]=summed_pe
        if self.initalization_finished==True:
            self.update_sim_2D_TRPES_bidir()
            
    @pyqtSlot('int')
    def on_comboBox__DisplayedComponent_bidir_pos_currentIndexChanged(self,value):
        self.comboBox_PE_current_bidir_pos.blockSignals(True)
        self.spinBox_PE_number_bidir_pos.blockSignals(True)
        self.spinBox_PE_number_bidir_pos.setValue(len(self.pe_gauss_bidir[0][value]))
        self.comboBox_PE_current_bidir_pos.clear()
        for i in range(len(self.pe_gauss_bidir[0][value])):
            self.comboBox_PE_current_bidir_pos.addItem(str(i+1))
        #self.load_pe_gauss_bidir(0)
        self.load_pe_gauss_bidir(0)
        self.comboBox_PE_current_bidir_pos.blockSignals(False)
        self.spinBox_PE_number_bidir_pos.blockSignals(False)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_pos,0)

    @pyqtSlot('int')
    def on_spinBox_PE_number_bidir_pos_valueChanged(self,value):
        #add a new one to the self.pe_gauss_bidir or delete it
        self.comboBox_PE_current_bidir_pos.blockSignals(True)
        if value>len(self.pe_gauss_bidir[0][self.comboBox__DisplayedComponent_bidir_pos.currentIndex()]):
            self.pe_gauss_bidir[0][self.comboBox__DisplayedComponent_bidir_pos.currentIndex()].append(skewed_gauss(0.,0.2,0.,1.))            
            #and update self.comboBox_PE_current
            self.comboBox_PE_current_bidir_pos.addItem(str(value))
        else:
            del self.pe_gauss_bidir[0][self.comboBox__DisplayedComponent_bidir_pos.currentIndex()][-1]
            #remove from combobox
            self.comboBox_PE_current_bidir_pos.removeItem(self.comboBox_PE_current_bidir_pos.count()-1)
        #put combobox and last index
        self.comboBox_PE_current_bidir_pos.setCurrentIndex(self.comboBox_PE_current_bidir_pos.count()-1)
        self.comboBox_PE_current_bidir_pos.blockSignals(False)
        self.load_pe_gauss_bidir(0)
        #update graphs etc
        self.update_pe_gauss_bidir(0)

    @pyqtSlot('int')
    def on_comboBox_PE_current_bidir_pos_currentIndexChanged(self,value):
        self.load_pe_gauss_bidir(0)

        
    def load_pe_gauss_bidir(self,n):
        #display the current gauss(or neutral)        
        index=[self.comboBox_PE_current_bidir_pos.currentIndex(),self.comboBox_PE_current_bidir_min.currentIndex()][n]
        displayed_component=[self.comboBox__DisplayedComponent_bidir_pos.currentIndex(),self.comboBox__DisplayedComponent_bidir_min.currentIndex()][n]
        current_gauss=self.pe_gauss_bidir[n][displayed_component][index]
        for i in [self.doubleSpinBox_PE_center_bidir_pos,self.doubleSpinBox_PE_center_bidir_min,
                  self.doubleSpinBox_PE_width_bidir_pos,self.doubleSpinBox_PE_width_bidir_min,
                  self.doubleSpinBox_pe_assymetry_bidir_pos,self.doubleSpinBox_pe_assymetry_bidir_min,
                  self.doubleSpinBox_pe_rel_intensity_bidir_pos,self.doubleSpinBox_pe_rel_intensity_bidir_min]:
            i.blockSignals(True)
        [self.doubleSpinBox_PE_center_bidir_pos,self.doubleSpinBox_PE_center_bidir_min][n].setValue(current_gauss.mu)
        [self.doubleSpinBox_PE_width_bidir_pos,self.doubleSpinBox_PE_width_bidir_min][n].setValue(current_gauss.fwhm)
        [self.doubleSpinBox_pe_assymetry_bidir_pos,self.doubleSpinBox_pe_assymetry_bidir_min][n].setValue(current_gauss.alpha)
        [self.doubleSpinBox_pe_rel_intensity_bidir_pos,self.doubleSpinBox_pe_rel_intensity_bidir_min][n].setValue(current_gauss.intensity)
        for i in [self.doubleSpinBox_PE_center_bidir_pos,self.doubleSpinBox_PE_center_bidir_min,
                  self.doubleSpinBox_PE_width_bidir_pos,self.doubleSpinBox_PE_width_bidir_min,
                  self.doubleSpinBox_pe_assymetry_bidir_pos,self.doubleSpinBox_pe_assymetry_bidir_min,
                  self.doubleSpinBox_pe_rel_intensity_bidir_pos,self.doubleSpinBox_pe_rel_intensity_bidir_min]:
            i.blockSignals(False)
        self.update_pe_viewer_bidir([self.viewer_pe_bidir_pos,self.viewer_pe_bidir_min][n],n)


    @pyqtSlot('double')
    def on_doubleSpinBox_PE_center_bidir_pos_valueChanged(self,value):
        self.update_pe_gauss_bidir(0)
    @pyqtSlot('double')
    def on_doubleSpinBox_PE_width_bidir_pos_valueChanged(self,value):
        self.update_pe_gauss_bidir(0)
    @pyqtSlot('double')
    def on_doubleSpinBox_pe_assymetry_bidir_pos_valueChanged(self,value):
        self.update_pe_gauss_bidir(0)
    @pyqtSlot('double')
    def on_doubleSpinBox_pe_rel_intensity_bidir_pos_valueChanged(self,value):
        self.update_pe_gauss_bidir(0)

    def update_pe_gauss_bidir(self,which):
        index=[self.comboBox_PE_current_bidir_pos.currentIndex(),self.comboBox_PE_current_bidir_min.currentIndex()][which]
        displayed_component=[self.comboBox__DisplayedComponent_bidir_pos.currentIndex(),self.comboBox__DisplayedComponent_bidir_min.currentIndex()][which]
        self.pe_gauss_bidir[which][displayed_component][index].mu=[self.doubleSpinBox_PE_center_bidir_pos.value(),
                                                                   self.doubleSpinBox_PE_center_bidir_min.value()][which]
        self.pe_gauss_bidir[which][displayed_component][index].fwhm=[self.doubleSpinBox_PE_width_bidir_pos.value(),
                                                                     self.doubleSpinBox_PE_width_bidir_min.value()][which]
        self.pe_gauss_bidir[which][displayed_component][index].alpha=[self.doubleSpinBox_pe_assymetry_bidir_pos.value(),
                                                                      self.doubleSpinBox_pe_assymetry_bidir_min.value()][which]
        self.pe_gauss_bidir[which][displayed_component][index].intensity=[self.doubleSpinBox_pe_rel_intensity_bidir_pos.value(),
                                                                           self.doubleSpinBox_pe_rel_intensity_bidir_min.value()][which]
        if which==0:
            self.update_pe_viewer_bidir(self.viewer_pe_bidir_pos,0)
        elif which==1:
            self.update_pe_viewer_bidir(self.viewer_pe_bidir_min,1)
        

    #----------------------------------
    #all functions regrouped from the tab time pose
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau1_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau2_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau3_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_1_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_2_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_3_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_irf_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_pos_irf_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_offset_bidir_pos_valueChanged(self,value):
        self.update_fit_params_bidir()

    @pyqtSlot('int')
    def on_comboBox_global_pick_model_bidir_pos_currentIndexChanged(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)   

    def update_fit_params_bidir(self):
        self.tabWidget.setUpdatesEnabled(False)
        #values for the max
        self.simulated_fit_bidir[0].tau1=self.doubleSpinBox_time_tau1_bidir_pos.value()
        self.simulated_fit_bidir[0].tau2=self.doubleSpinBox_time_tau2_bidir_pos.value()
        self.simulated_fit_bidir[0].tau3=self.doubleSpinBox_time_tau3_bidir_pos.value()
        self.simulated_fit_bidir[0].sigma_1=[self.doubleSpinBox_time_int_1_bidir_pos.value()]
        self.simulated_fit_bidir[0].sigma_2=[self.doubleSpinBox_time_int_2_bidir_pos.value()]
        self.simulated_fit_bidir[0].sigma_3=[self.doubleSpinBox_time_int_3_bidir_pos.value()]
        self.simulated_fit_bidir[0].sigma_offset=[self.doubleSpinBox_time_int_offset_bidir_pos.value()]
        self.simulated_fit_bidir[0].fwhm=[self.doubleSpinBox_time_irf_bidir_pos.value()]
        self.simulated_fit_bidir[0].time_offset=[self.doubleSpinBox_time_pos_irf_bidir_pos.value()]
        
        #values for the min
        self.simulated_fit_bidir[1].tau1=self.doubleSpinBox_time_tau1_bidir_min.value()
        self.simulated_fit_bidir[1].tau2=self.doubleSpinBox_time_tau2_bidir_min.value()
        self.simulated_fit_bidir[1].tau3=self.doubleSpinBox_time_tau3_bidir_min.value()
        self.simulated_fit_bidir[1].sigma_1=[self.doubleSpinBox_time_int_1_bidir_min.value()]
        self.simulated_fit_bidir[1].sigma_2=[self.doubleSpinBox_time_int_2_bidir_min.value()]
        self.simulated_fit_bidir[1].sigma_3=[self.doubleSpinBox_time_int_3_bidir_min.value()]
        self.simulated_fit_bidir[1].sigma_offset=[self.doubleSpinBox_time_int_offset_bidir_min.value()]
        self.simulated_fit_bidir[1].fwhm=[self.doubleSpinBox_time_irf_bidir_pos.value()]
        self.simulated_fit_bidir[1].time_offset=[self.doubleSpinBox_time_pos_irf_bidir_pos.value()]
        if self.initalization_finished==True:
            self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)   
            self.update_time_graph_bidir(self.viewer_time_bidir_min,1)   
        self.tabWidget.setUpdatesEnabled(True)
    
    @pyqtSlot('double')
    def on_doubleSpinBox_time_min_bidir_pos_valueChanged(self,value):
        self.doubleSpinBox_time_min_bidir_pos.blockSignals(True)
        if value>self.doubleSpinBox_time_max_bidir_pos.value():
            self.doubleSpinBox_time_min_bidir_pos.setValue(self.doubleSpinBox_time_max_bdir_pos.value()-1)
        self.doubleSpinBox_time_min_bidir_pos.blockSignals(False)
        self.update_time_fit_range_bidir(which='pos')

    @pyqtSlot('double')
    def on_doubleSpinBox_time_max_bidir_pos_valueChanged(self,value):
        self.doubleSpinBox_time_max_bidir_pos.blockSignals(True)
        if value<self.doubleSpinBox_time_min_bidir_pos.value():
            self.doubleSpinBox_time_max_bidir_pos.setValue(self.doubleSpinBox_time_min_bidir_pos.value()+1)
        self.doubleSpinBox_time_max_bidir_pos.blockSignals(False)
        self.update_time_fit_range_bidir(which='pos')
    
    def update_time_fit_range_bidir(self,which='pos'):
        for i in [self.doubleSpinBox_time_min_bidir_min,self.doubleSpinBox_time_min_bidir_pos,self.doubleSpinBox_time_max_bidir_min,self.doubleSpinBox_time_max_bidir_pos]:
            i.blockSignals(True)
        if which=='pos':
            #change min values to the pos values
            self.doubleSpinBox_time_min_bidir_min.setValue(self.doubleSpinBox_time_min_bidir_pos.value())
            self.doubleSpinBox_time_max_bidir_min.setValue(self.doubleSpinBox_time_max_bidir_pos.value())
        elif which=='min':
            #change pos values to the min values
            self.doubleSpinBox_time_min_bidir_pos.setValue(self.doubleSpinBox_time_min_bidir_min.value())
            self.doubleSpinBox_time_max_bidir_pos.setValue(self.doubleSpinBox_time_max_bidir_min.value())
        #update the self.time_fit_range_bidir
        stepsize=(self.doubleSpinBox_time_max_bidir_pos.value()-self.doubleSpinBox_time_min_bidir_pos.value())/200
        self.time_fit_range_bidir=np.arange(self.doubleSpinBox_time_min_bidir_pos.value(),self.doubleSpinBox_time_max_bidir_pos.value(),stepsize)
        self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)
        self.update_time_graph_bidir(self.viewer_time_bidir_min,1)        
        for i in [self.doubleSpinBox_time_min_bidir_min,self.doubleSpinBox_time_min_bidir_pos,self.doubleSpinBox_time_max_bidir_min,self.doubleSpinBox_time_max_bidir_pos]:
            i.blockSignals(False)

    @pyqtSlot('bool')
    def on_radioButton_no_offset_bidir_pos_toggled(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_bidir_pos_toggled(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_pos,0)
        
    def update_time_graph_bidir(self,viewer,n):
        """
        updating the time plot
        """
        #calculating the necessary curves
        self.time_curves_bidir[n]=[]
        t=self.time_fit_range_bidir+self.simulated_fit_bidir[n].time_offset
        if n==1:
            t=-t#switch it to fit the negative decay            
        all_curves_taus=[self.simulated_fit_bidir[n].mono_exp_decay(t,0),
                    self.simulated_fit_bidir[n].bi_exp_decay_population(t,0),
                    self.simulated_fit_bidir[n].tri_exp_decay_population(t,0)]
        all_curves_final_offset=[self.simulated_fit_bidir[n].mono_exp_decay_final_state_pop(t,0),
                                 self.simulated_fit_bidir[n].bi_exp_decay_final_state_pop(t,0),
                                 self.simulated_fit_bidir[n].tri_exp_decay_final_state_pop(t,0)]
        all_curves_offset=self.simulated_fit_bidir[n].offset(t,0)
        model_chosen=[self.comboBox_global_pick_model_bidir_pos.currentIndex()+1,self.comboBox_global_pick_model_bidir_min.currentIndex()+1][n]
        names1=['mono-exp','bi-exp','tri-exp']
        self.time_curves_bidir[n]=all_curves_taus[:model_chosen]
        names=names1[:model_chosen]
        final_offset=[self.radioButton_time_final_offset_bidir_pos.isChecked(),self.radioButton_time_final_offset_bidir_min.isChecked()]
        offset=[self.radioButton_time_offset_bidir_pos.isChecked(),self.radioButton_time_offset_bidir_min.isChecked()]
        if final_offset[n]==True:
            self.time_curves_bidir[n].append(all_curves_final_offset[model_chosen-1])
            names.append('offset')
        elif offset[n]==True:
            self.time_curves_bidir[n].append(all_curves_offset)
            names.append('offset')
        legend=self.bidir_legends_time[n]
        #delete the old legend
        """
        try:
            legend.scene().removeItem(legend)
        except Exception as e:
           pass  
        """
        self.bidir_legends_time[n]=viewer.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        #make a sum curve
        summed_decay=np.zeros(self.time_fit_range_bidir.shape)
        for curve in self.time_curves_bidir[n]:
            summed_decay+=curve
        #plot that stuff
        viewer.clear()
        
        for i,curve in enumerate(self.time_curves_bidir[n]):
            viewer.plot(self.time_fit_range_bidir,0.8*curve,
                                pen=pg.mkPen(self.colors[i],width=2),name=names[i])
        viewer.plot(self.time_fit_range_bidir,summed_decay,pen=pg.mkPen((0,0,0),width=2))
        if self.initalization_finished==True:
            self.update_sim_2D_TRPES_bidir()
        
    #----------------------------------
    #all functions regrouped from the tab PE min
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_max_Energy_PE_bidir_min_valueChanged(self,value):
        self.update_pe_energy_range_bidir(value,'min')

    @pyqtSlot('int')
    def on_comboBox__DisplayedComponent_bidir_min_currentIndexChanged(self,value):
        self.spinBox_PE_number_bidir_min.blockSignals(True)
        self.comboBox_PE_current_bidir_min.blockSignals(True)
        self.spinBox_PE_number_bidir_min.setValue(len(self.pe_gauss_bidir[1][value]))
        self.comboBox_PE_current_bidir_min.clear()
        for i in range(len(self.pe_gauss_bidir[1][value])):
            self.comboBox_PE_current_bidir_min.addItem(str(i+1))
        #self.load_pe_gauss_bidir(0)
        self.load_pe_gauss_bidir(1)
        self.spinBox_PE_number_bidir_min.blockSignals(False)
        self.comboBox_PE_current_bidir_min.blockSignals(False)
        self.update_pe_viewer_bidir(self.viewer_pe_bidir_min,1)

    @pyqtSlot('int')
    def on_spinBox_PE_number_bidir_min_valueChanged(self,value):
        #add a new one to the self.pe_gauss_bidir or delete it
        self.comboBox_PE_current_bidir_min.blockSignals(True)
        if value>len(self.pe_gauss_bidir[1][self.comboBox__DisplayedComponent_bidir_min.currentIndex()]):
            self.pe_gauss_bidir[1][self.comboBox__DisplayedComponent_bidir_min.currentIndex()].append(skewed_gauss(0.,0.2,0.,1.))            
            #and update self.comboBox_PE_current
            self.comboBox_PE_current_bidir_min.addItem(str(value))
        else:
            del self.pe_gauss_bidir[1][self.comboBox__DisplayedComponent_bidir_min.currentIndex()][-1]
            #remove from combobox
            self.comboBox_PE_current_bidir_min.removeItem(self.comboBox_PE_current_bidir_min.count()-1)
        #put combobox and last index
        self.comboBox_PE_current_bidir_min.setCurrentIndex(self.comboBox_PE_current_bidir_min.count()-1)
        self.comboBox_PE_current_bidir_min.blockSignals(False)
        self.load_pe_gauss_bidir(1)
        #update graphs etc
        self.update_pe_gauss_bidir()

    @pyqtSlot('double')
    def on_doubleSpinBox_PE_center_bidir_min_valueChanged(self,value):
        self.update_pe_gauss_bidir(1)
    @pyqtSlot('double')
    def on_doubleSpinBox_PE_width_bidir_min_valueChanged(self,value):
        self.update_pe_gauss_bidir(1)
    @pyqtSlot('double')
    def on_doubleSpinBox_pe_assymetry_bidir_min_valueChanged(self,value):
        self.update_pe_gauss_bidir(1)
    @pyqtSlot('double')
    def on_doubleSpinBox_pe_rel_intensity_bidir_min_valueChanged(self,value):
        self.update_pe_gauss_bidir(1)
    @pyqtSlot('int')
    def on_comboBox_PE_current_bidir_min_currentIndexChanged(self,value):
        self.load_pe_gauss_bidir(1)
    #----------------------------------
    #all functions regrouped from the tab time min
    #----------------------------------
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau1_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau2_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_tau3_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_1_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_2_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_3_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_time_int_offset_bidir_min_valueChanged(self,value):
        self.update_fit_params_bidir()
        
    @pyqtSlot('int')
    def on_comboBox_global_pick_model_bidir_min_currentIndexChanged(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_min,1)   
        
    @pyqtSlot('double')
    def on_doubleSpinBox_time_min_bidir_min_valueChanged(self,value):
        self.doubleSpinBox_time_min_bdir_min.blockSignals(True)
        if value>self.doubleSpinBox_time_max_bdir_min.value():
            self.doubleSpinBox_time_min_bdir_min.setValue(self.doubleSpinBox_time_max_bdir_min.value()-1)
        self.tabWidget.setUpdatesEnabled(True)
        self.doubleSpinBox_time_min_bdir_min.blockSignals(False)
        self.update_time_fit_range_bidir(which='min')
    @pyqtSlot('double')
    def on_doubleSpinBox_time_max_bidir_min_valueChanged(self,value):
        self.doubleSpinBox_time_max_bidir_min.blockSignals(True)
        if value<self.doubleSpinBox_time_min_bidir_min.value():
            self.doubleSpinBox_time_max_bidir_min.setValue(self.doubleSpinBox_time_min_bidir_min.value()+1)
        self.doubleSpinBox_time_max_bidir_min.blockSignals(False)
        self.update_time_fit_range_bidir(which='min')

    @pyqtSlot('bool')
    def on_radioButton_no_offset_bidir_min_toggled(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_min,1)
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_bidir_min_toggled(self,value):
        self.update_time_graph_bidir(self.viewer_time_bidir_min,1)
    #----------------------------------
    #all functions regrouped from the TRPES and miscancellous
    #----------------------------------
    def update_sim_2D_TRPES_bidir(self):
        #calculate the 2D TRPES from the considered components, pe &time
        ones_matrix=np.ones((self.pe_energy_range_bidir.shape[0],self.time_fit_range_bidir.shape[0]))
        #print('pe',self.pe_energy_range.shape,'time',self.time_fit_range.shape)
        self.sim_2D_TRPES_bidir=np.zeros((self.pe_energy_range_bidir.shape[0],self.time_fit_range_bidir.shape[0]))
        #sum the pos side
        for i in range(self.comboBox_global_pick_model_bidir_pos.currentIndex()+1):
            intermed=np.reshape(self.pe_spectra_bidir[0][i],(self.pe_energy_range_bidir.shape[0],1))*ones_matrix*self.time_curves_bidir[0][i]
            self.sim_2D_TRPES_bidir+=intermed
        if self.radioButton_no_offset_bidir_pos.isChecked()==False:
            intermed=np.reshape(self.pe_spectra_bidir[0][-1],(self.pe_energy_range_bidir.shape[0],1))*ones_matrix*self.time_curves_bidir[0][-1]
            self.sim_2D_TRPES_bidir+=intermed
        #sum the min side
        for i in range(self.comboBox_global_pick_model_bidir_min.currentIndex()+1):
            intermed=np.reshape(self.pe_spectra_bidir[1][i],(self.pe_energy_range_bidir.shape[0],1))*ones_matrix*self.time_curves_bidir[1][i]
            self.sim_2D_TRPES_bidir+=intermed
        if self.radioButton_no_offset_bidir_min.isChecked()==False:
            intermed=np.reshape(self.pe_spectra_bidir[1][-1],(self.pe_energy_range_bidir.shape[0],1))*ones_matrix*self.time_curves_bidir[1][-1]
            self.sim_2D_TRPES_bidir+=intermed
        self.sim_2D_TRPES_bidir=self.sim_2D_TRPES_bidir.transpose()
        #add the noise if the user desired it
        if self.checkBox_random_noise_bidir.isChecked()==True:
            noise = np.random.normal(0, self.doubleSpinBox_noise_bidir.value(), self.sim_2D_TRPES_bidir.shape)
            self.sim_2D_TRPES_bidir+=noise
        self.plot_TRPES(self.viewer_sim_trpes_bidir,self.sim_2D_TRPES_bidir,self.time_fit_range_bidir,self.pe_energy_range_bidir)
 
    @pyqtSlot('bool')
    def on_checkBox_random_noise_bidir_toggled(self):
        self.update_sim_2D_TRPES_bidir()
    @pyqtSlot('double')
    def on_doubleSpinBox_noise_bidir_valueChanged(self,value):
        if self.checkBox_random_noise_bidir.isChecked()==True:
            self.update_sim_2D_TRPES_bidir()
        
    ##########################
    #all functions for SVD analysis
    ##########################
    @pyqtSlot()
    def on_pushButton_get_simulated_TRPES_bidir_clicked(self):
        self.SVD_TRPES=self.sim_2D_TRPES_bidir
        self.SVD_time=self.time_fit_range_bidir
        self.SVD_eV=self.pe_energy_range_bidir
        #plot it
        self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
        #go on to subfunction to make the SVD analysis and plot the rest
        self.SVD_analysis()
    
    @pyqtSlot()
    def on_pushButton_get_simulated_TRPES_clicked(self):
        '''
        transfer the simulated TRPES from the other tab to this one here
        '''
        self.SVD_TRPES=self.sim_2D_TRPES
        self.SVD_time=self.time_fit_range
        self.SVD_eV=self.pe_energy_range
        #plot it
        self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
        #go on to subfunction to make the SVD analysis and plot the rest
        self.SVD_analysis()
        
    @pyqtSlot()
    def on_pushButton_load_TRPES_clicked(self):
        '''
        load the TRPES file from some other file, either csv file (andrey format), or one of mine or simple 2D matrix.
        '''
        filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
        if filename!='':
            self.dir=filename[:(-len(filename.split('/')[-1]))]
            if filename[-3:]=='csv':
                #load the data as a csv file
                self.SVD_time,self.SVD_eV,self.SVD_TRPES=self.load_csv(filename)
                #self.SVD_TRPES=self.SVD_TRPES.transpose()
                self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
                self.SVD_analysis()
            elif filename!='':
                #load my old data format
                loaded=np.loadtxt(filename,comments='# ')
                self.SVD_time,self.SVD_eV,self.SVD_TRPES=self.convert_data_to_plottable(loaded)
                self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
                self.SVD_analysis()
                
    def load_csv(self,filename):
        data=np.loadtxt(filename,delimiter=',')
        t=data[1:,0]
        eV=data[0,1:]
        print(t[0],eV[0])
        z=data[1:,1:]/data[1:,1:].max()
        if t[0]>t[-1]:
            z=np.flip(z,axis=0)
            t=np.flip(t,axis=0)
        print('first value of x:',t[0])
        return t,eV,z
    
    @pyqtSlot()
    def on_pushButton_Save_TRPES_clicked(self):
        """
        save in the .csv format
        """
        if type(self.SVD_TRPES)!=type(None):
            #make one bigg array
            to_save=np.zeros((self.SVD_TRPES.shape[0]+1,self.SVD_TRPES.shape[1]+1))
            reply=QMessageBox.question(self, 'Save which one?',
                                       'Save left(original, click yes) or right(reconstructed or residual, whatever selected, click no) graph?'
                                       )#StandardButtons=(QMessageBox.Yes, QMessageBox.No))
            if reply==QMessageBox.Yes:
                to_save[1:,1:]=self.SVD_TRPES
            if reply== QMessageBox.No:
                reconstructed=np.zeros(self.SVD_TRPES.shape)
                for i in range(self.spinBox_SVD_which_component.value()):
                    reconstructed+=np.reshape(self.SVD_time_rec[:,i],(self.SVD_time.shape[0],1))*(self.SVD_values[i]*np.ones(self.SVD_TRPES.shape))*self.SVD_eV_rec[i,:]
                if self.comboBox_SVD_show_residual.currentIndex()==1:
                    reconstructed=self.SVD_TRPES-reconstructed
                to_save[1:,1:]=reconstructed
            to_save[1:,0]=self.SVD_time
            to_save[0,1:]=self.SVD_eV
            filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
            filename+='.csv'
            np.savetxt(filename,to_save,delimiter=',')
        
    def convert_data_to_plottable(self,data):
        """
        old function used to load data from lionel-like formats
        """
        try:
            # if it was saved with this program, it should work:
            Matrix=data
            print(Matrix.shape)
            x=[]
            y=[]
            z=[]
            g=0
            for i in range(Matrix.shape[0]):
                if i==0:
                    x.append(Matrix[i,0])
                    y.append(Matrix[i,1])
                    z.append(Matrix[i,2])
                elif Matrix[i,0]>=Matrix[(i-1),0]:
                    if Matrix[i,0] not in x:
                        x.append(Matrix[i,0])
                    if Matrix[i,1] not in y:
                        y.append(Matrix[i,1])
                    
                    z.append(Matrix[i,2])
                else:
                    z.append([])
                    
                    g=g+1
            #shape the matrix z into the correct shape
            z=np.array(z)
            z=np.transpose(np.reshape(z,(len(x),len(y))))
            z=z/z.max()
        except:
            #print 'did not work,tkinter file?'
            # apparantly saved with old tkinter version, so do:
            Matrix=data
            x=[]
            y=[]
            z=[]
            intermed=[]
            for i in range(1,Matrix.shape[0]):
                if Matrix[i,0]>=Matrix[(i-1),0]:
                    if Matrix[i,0] not in x:
                        x.append(Matrix[i,0])
                        intermed=[]
                    if Matrix[i,1] not in y and Matrix[i,1]!=0.0:
                        y.append(Matrix[i,1])
                    if Matrix[i,2]!=0.0 and Matrix[i,1]!=0 and Matrix[i,1] not in intermed:
                        z.append(Matrix[i,2])
                        intermed.append(Matrix[i,1])
            z=np.array(z)
            z=np.transpose(np.reshape(z,(len(x),len(y))))
            z=z/z.max()
        return(np.array(x)/1000.,np.array(y),z.transpose())
    

        
    def SVD_analysis(self):
        """
        core function to make the SVD analysis and plot all necessary stuff
        """
        self.SVD_time_rec,self.SVD_values,self.SVD_eV_rec=np.linalg.svd(self.SVD_TRPES,full_matrices=True)
        #put it on reconstructed TRPES
        self.comboBox_SVD_show_residual.setCurrentIndex(0)
        self.on_comboBox_SVD_show_residual_currentIndexChanged(0)
        self.spinBox_SVD_which_component.setValue(1)
        self.on_spinBox_SVD_which_component_valueChanged(1)
        #update SVD weights aka the overview table
        self.update_SVD_weights()
        
    @pyqtSlot('bool')
    def on_checkBox_global_subtract_background_svd_toggled(self, toggled):
        '''
        subtract background, but keep the original file. Make therefore a transfer file.
        '''
        if type(self.SVD_TRPES)!=type(None):
            if type(self.SVD_original_for_bg)==type(None):
                self.SVD_original_for_bg=copy.deepcopy(self.SVD_TRPES)
            if toggled==False:
                #revert back to no background subtraction
                self.SVD_TRPES=copy.deepcopy(self.SVD_original_for_bg)
            else:
                self.SVD_TRPES=self.subtract_background(self.SVD_original_for_bg,self.SVD_time,
                                                           self.doubleSpinBox_global_bg_subtract_from_svd.value(),
                                                           self.doubleSpinBox_global_bg_subtract_to_svd.value())
            self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
            self.SVD_analysis()

    def find_nearest_index(self,array,value):
        '''
        returns index of the position in the array whose values is closest to value
        '''
        return np.abs(array - value).argmin()

    @pyqtSlot('bool')
    def on_actionSwitch_TRPES_axis_toggled(self,value):
        self.switch_TRPES_axis=value
        #replot all
        #The simulated stuff
        self.plot_TRPES(self.viewer_sim_trpes,self.sim_2D_TRPES,self.time_fit_range,self.pe_energy_range)
        self.plot_TRPES(self.viewer_sim_trpes_bidir,self.sim_2D_TRPES_bidir,self.time_fit_range_bidir,self.pe_energy_range_bidir)
        #SVD
        if type(self.SVD_TRPES)!=type(None):
            self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
            self.on_comboBox_SVD_show_residual_currentIndexChanged(self.comboBox_SVD_show_residual.currentIndex())
        #Global 1:
        self.on_comboBox_global_TRPES_which_currentIndexChanged(self.comboBox_global_TRPES_which.currentIndex())
        self.on_comboBox_global_fit_or_residual_currentIndexChanged(self.comboBox_global_fit_or_residual.currentIndex())
        #Global 2:
        if type(self.global_TRPES_bidir)!=type(None):
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
        self.on_comboBox_global_fit_or_residual_bidir_currentIndexChanged(self.comboBox_global_fit_or_residual_bidir.currentIndex())
            
        
    def change_viewer_scale_to_log(self,value):
        print('I called this one',value)
        if value==True:
            self.display_log_t=False
        else:
            self.display_log_t=True
        #replot all
        #The simulated stuff
        self.plot_TRPES(self.viewer_sim_trpes,self.sim_2D_TRPES,self.time_fit_range,self.pe_energy_range)
        self.plot_TRPES(self.viewer_sim_trpes_bidir,self.sim_2D_TRPES_bidir,self.time_fit_range_bidir,self.pe_energy_range_bidir)
        #SVD
        if type(self.SVD_TRPES)!=type(None):
            self.plot_TRPES(self.viewer_sim_trpes_SVD,self.SVD_TRPES,self.SVD_time,self.SVD_eV)
            self.on_comboBox_SVD_show_residual_currentIndexChanged(self.comboBox_SVD_show_residual.currentIndex())
        #Global 1:
        self.on_comboBox_global_TRPES_which_currentIndexChanged(self.comboBox_global_TRPES_which.currentIndex())
        self.on_comboBox_global_fit_or_residual_currentIndexChanged(self.comboBox_global_fit_or_residual.currentIndex())
        #Global 2:
        if type(self.global_TRPES_bidir)!=type(None):
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
        self.on_comboBox_global_fit_or_residual_bidir_currentIndexChanged(self.comboBox_global_fit_or_residual_bidir.currentIndex())
        
 
    def plot_TRPES(self, viewer,z,x,y):
        """
        function to plot the TRPES on an appropriate viewer, need to implement it once I know how to properly display such stuff
        viewer: the plotwidget
        z: np 2D array of dimensions [M,N]
        y: 1D array of dimensions N
        x: 1D array of dimensions M
        """
        '''
        self.D2Trpes = pg.ImageView()
        self.D2Trpes.setImage(z)
        viewer.addItem(self.D2Trpes)
        '''
        try:
            if self.display_log_t==True:
                f = interpolate.interp2d(x, y, z.transpose(), kind='linear')
                xi1=-np.flip(np.logspace(np.log10(0.1),np.log10(-x.min()),250,endpoint=True),axis=0)
                xi2=np.logspace(np.log10(0.1),np.log10(x.max()),250,endpoint=True)
                xi=np.concatenate([xi1,xi2],axis=0)
                yi=np.linspace(y.min(),y.max()+np.abs(y[0]-y[1]),500)
                print('shapes',xi.shape,yi.shape)
                z2=f(xi,yi).transpose()
                if self.switch_TRPES_axis==True:
                    y3=xi
                    x3=yi
                    z2=z2.transpose()
                else:
                    y3=yi
                    x3=xi
                #alright, clear the viewer and add that stuff
                viewer.clear()
                p1 = viewer.addPlot()
                self.filterMenu = QtGui.QMenu("Logharithmic Scale")
                self.change_to_logharithmic = QtGui.QAction("Change x-axis to logarithmic", self.filterMenu,checkable=True,checked=self.display_log_t)
                self.change_to_logharithmic.triggered.connect(lambda: self.change_viewer_scale_to_log(self.change_to_logharithmic.isChecked()))
                self.filterMenu.addAction(self.change_to_logharithmic)
                p1.ctrlMenu=[self.filterMenu]
                #add the colormap
                colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
                #color=np.array([[0,0,0,255],[255,128,0,255],[255,255,0,255]],dtype=np.ubyte)
                colormap._init()
                color=np.array((colormap._lut * 255).view(np.ndarray)[:-4,:],dtype=np.ubyte)
                pos=np.array(np.arange(0.,1.,1./color.shape[0]))
                map=pg.ColorMap(pos,color)
                lut=map.getLookupTable(0.,1.,256)
                # Item for displaying image data
                img = pg.ImageItem()
                p1.addItem(img)
                img.setImage(z2)
                img.setLookupTable(lut)
                img.setLevels([0,z2.max()])
                if self.switch_TRPES_axis==False:
                    p1.setLabel('left', "energy", units='eV')
                    p1.setLabel('bottom', "log(time)", units='ps')
                else:
                    p1.setLabel('left', "log(time)", units='ps')
                    p1.setLabel('bottom', "energy", units='eV')
                    
                     # Contrast/color control
                hist = pg.HistogramLUTItem()
                hist.setImageItem(img)
                viewer.addItem(hist)
        
                # Draggable line for setting isocurve level
                isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
                hist.vb.addItem(isoLine)
                hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
                isoLine.setValue(0.)
                isoLine.setZValue(z2.max()) # bring iso line above contrast controls
                hist.gradient.setColorMap(map)
        
                img.setLookupTable(lut)
                img.setLevels([0,z2.max()])
        
                if self.switch_TRPES_axis==False:
                    size_y=np.abs(y3[-1]-y3[0])
                    y_factor=size_y/y3.shape[0]
                    img.translate(0., y3.min())
                    img.scale(1., y_factor) 
                    
                    
                    xaxis=p1.getAxis('bottom')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(x3,range(len(x3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks) 
                else:
                    size=np.abs(x3[-1]-x3[0])
                    x_factor=size/x3.shape[0]
                    img.translate(x3.min(),0.)
                    img.scale(x_factor,1) 
                    
                    
                    xaxis=p1.getAxis('left')
        
                    values=[]
                    strings=[]
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((-10**(i+1)))
                                strings.append('-10^'+str(i+1))
                            else:
                                values.append(-j*(10**i))
                                strings.append('')
                    values.append(0)
                    strings.append('0')
                    for i in [-1,0,1,2,3,4]:
                        for j in [1,2,3,4,5,6,7,8,9,10]:
                            if j==10:
                                values.append((10**(i+1)))
                                strings.append('10^'+str(i+1))
                            else:
                                values.append(j*(10**i))
                                strings.append('')
                    ticks=[list(zip(values,strings))]
                    #xaxis.setTicks(ticks)    
        
                    #second try:
                    f2=interpolate.interp1d(y3,range(len(y3)),  fill_value='extrapolate') 
                    y4=f2(np.array(values))
                    y4=list(y4)
                    ticks=[list(zip(y4,strings))]
                    xaxis.setTicks(ticks)
    
            else:
                self.plot_TRPES_normally(viewer, z,x,y)
        except AttributeError:
            self.plot_TRPES_normally(viewer, z,x,y)
    
            

    def plot_TRPES_normally(self, viewer, z,x,y):
        z_orig=copy.deepcopy(z)
        #test whether x is equally spaced
        x2=x[:-1]-x[1:]
        y2=y[:-1]-y[1:]
              
        if np.any(np.abs(x2-x2[0])>0.0001)==True or np.any(np.abs(x2-x2[0])>0.001)==True:
            print('interpolating')
            f = interpolate.interp2d(x, y, z.transpose(), kind='linear')
            xi=np.arange(x.min(),x.max(),np.abs(x2).min())
            yi=np.arange(y.min(),y.max()+np.abs(y2[0]),np.abs(y2).min())
            z=f(xi,yi).transpose()
        else:
            xi=x
            yi=y
        if self.switch_TRPES_axis==True:
            y3=xi
            x3=yi
            z=z.transpose()
        else:
            y3=yi
            x3=xi
    
        # A plot area (ViewBox + axes) for displaying the image
        viewer.clear()
        p1 = viewer.addPlot()
        self.filterMenu = QtGui.QMenu("Logharithmic Scale")
        self.change_to_logharithmic = QtGui.QAction("Change x-axis to logarithmic", self.filterMenu,checkable=True,checked=self.display_log_t)
        self.change_to_logharithmic.triggered.connect(lambda: self.change_viewer_scale_to_log(self.change_to_logharithmic.isChecked()))
        self.filterMenu.addAction(self.change_to_logharithmic)
        p1.ctrlMenu=[self.filterMenu]
        #pos=np.array([0.,0.5,1.0])
        colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
        #color=np.array([[0,0,0,255],[255,128,0,255],[255,255,0,255]],dtype=np.ubyte)
        colormap._init()
        color=np.array((colormap._lut * 255).view(np.ndarray)[:-4,:],dtype=np.ubyte)
        pos=np.array(np.arange(0.,1.,1./color.shape[0]))
        map=pg.ColorMap(pos,color)
        lut=map.getLookupTable(0.,1.,256)
        # Item for displaying image data
        img = pg.ImageItem()
        p1.addItem(img)
        img.setImage(z)
        img.setLookupTable(lut)
        img.setLevels([0,z.max()])
        if self.switch_TRPES_axis==False:
            p1.setLabel('left', "energy", units='eV')
            p1.setLabel('bottom', "time", units='ps')
        else:
            p1.setLabel('left', "time", units='ps')
            p1.setLabel('bottom', "energy", units='eV')
        

        # Contrast/color control
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        viewer.addItem(hist)

        # Draggable line for setting isocurve level
        isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        hist.vb.addItem(isoLine)
        hist.vb.setMouseEnabled(y=False) # makes user interaction a little easier
        isoLine.setValue(0.)
        isoLine.setZValue(z.max()) # bring iso line above contrast controls
        hist.gradient.setColorMap(map)

        img.setLookupTable(lut)
        img.setLevels([0,z.max()])

        # set position and scale of image
        size=np.abs(x3[-1]-x3[0])
        x_factor=size/x3.shape[0]

        size_y=np.abs(y3[-1]-y3[0])
        y_factor=size_y/y3.shape[0]
        img.translate(x3.min(), y3.min())
        img.scale(x_factor, y_factor) 
        p1.setXRange(x3[0],x3[-1])
        p1.setYRange(y3[0],y3[-1])
        
    
    @pyqtSlot('int')
    def on_comboBox_SVD_show_residual_currentIndexChanged(self,value):
        """
        update the viewer_sim_trpes_SVD_simulated
        """
        if type(self.SVD_TRPES)!=type(None):
            reconstructed=np.zeros(self.SVD_TRPES.shape)
            for i in range(self.spinBox_SVD_which_component.value()):
                #reconstructed+=np.reshape(self.SVD_eV_rec[i,:],(self.SVD_eV.shape[0],1))*(self.SVD_values[i]*np.ones(self.SVD_TRPES.shape))*self.SVD_time_rec[i,:]
                reconstructed+=np.reshape(self.SVD_time_rec[:,i],(self.SVD_time.shape[0],1))*(self.SVD_values[i]*np.ones(self.SVD_TRPES.shape))*self.SVD_eV_rec[i,:]
                self.SVD_reconstructed=reconstructed
            if value==1:
                reconstructed=self.SVD_TRPES-reconstructed        
            self.plot_TRPES(self.viewer_sim_trpes_SVD_simulated,reconstructed,self.SVD_time,self.SVD_eV)

    
    @pyqtSlot('int')
    def on_spinBox_SVD_which_component_valueChanged(self,value):
        """
        update both the viewer_SVD_time and the viewer_SVD_energy with as many components as value
        """
        if type(self.SVD_TRPES)!=type(None):
            #update the viewer_SVD_time
            self.viewer_SVD_time.clear()
            """
            #delete the old legend
            try:
                self.viewer_SVD_time_legend.scene().removeItem(self.viewer_SVD_time_legend)
            except Exception as e:
               pass
            """
            self.viewer_SVD_time_legend=self.viewer_SVD_time.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
            for i in range(value):
                curve=-self.SVD_time_rec[:,i]
                self.viewer_SVD_time.plot(self.SVD_time,curve,
                                    pen=pg.mkPen(self.colors[i],width=2),name=str(i)) 
            #update the viewer_SVD_energy 
            self.viewer_SVD_energy.clear()
            """
            #delete the old legend
            try:
                self.viewer_SVD_pe_legend.scene().removeItem(self.viewer_SVD_pe_legend)
            except Exception as e:
               pass
            """
            self.viewer_SVD_pe_legend=self.viewer_SVD_energy.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
            for i in range(value):
                curve=-self.SVD_eV_rec[i,:]
                self.viewer_SVD_energy.plot(self.SVD_eV,curve,
                                    pen=pg.mkPen(self.colors[i],width=2),name=str(i))
            #update the SVD viewer
            self.on_comboBox_SVD_show_residual_currentIndexChanged(self.comboBox_SVD_show_residual.currentIndex())
        
    def update_SVD_weights(self):
        text= 'weight | autocorr\n'
        text+='-----------------\n'
        limit_reached=False
        for n in range(20):
            #calculate the autocorr factor
            autocorr=0
            for i in range(1,self.SVD_eV_rec.shape[0]-1):
                #not sure if here I should not switch n and i
                autocorr+=self.SVD_eV_rec[n,i]*self.SVD_eV_rec[n,i+1]
            if autocorr<0.5 and limit_reached==False:
                limit_reached=True
                text+='-----------------\n'
            text+=" {:.2f}".format(self.SVD_values[n])+'     |    '+"{:.2f}".format(autocorr)+'\n'
        self.textBrowser_SVD_overview.setText(text)
    
    def initialize_SVD_graphs(self): 
        self.viewer_SVD_time.setLabel('left', "intensity", units='arb.u.')
        self.viewer_SVD_time.setLabel('bottom', "time", units='ps')
        self.viewer_SVD_time.plotItem.ctrlMenu=[]
        self.viewer_SVD_energy.setLabel('left', "intensity", units='arb.u.')
        self.viewer_SVD_energy.setLabel('bottom', "energy", units='eV')
        self.viewer_SVD_energy.plotItem.ctrlMenu=[]

  
    ##########################
    #all functions for the global analysis 
    ##########################

    def initialize_global_graphs(self):
        self.viewer_global_time_guess.setTitle('Total Photoelectron decay')
        self.viewer_global_time_guess.setLabel('left', "intensity", units='arb.u.')
        self.viewer_global_time_guess.setLabel('bottom', "time", units='ps')
        self.viewer_global_pe_guess.setLabel('left', "intensity", units='arb.u.')
        self.viewer_global_pe_guess.setLabel('bottom', "energy", units='eV')
        self.viewer_DAS.setLabel('left', "intensity", units='arb.u.')
        self.viewer_DAS.setLabel('bottom', "energy", units='eV')
        self.filterMenu2 = QtGui.QMenu("Normalize")
        self.normalize = QtGui.QAction("Normalize the DAS", self.filterMenu2,checkable=True,checked=False)
        self.normalize.triggered.connect(lambda: self.normalize_viewer_plot_DAS('hello'))
        self.filterMenu2.addAction(self.normalize)
        self.viewer_DAS.plotItem.ctrlMenu=[self.filterMenu2]
        self.viewer_DAS_decays.setLabel('left', "intensity", units='arb.u.')
        self.viewer_DAS_decays.setLabel('bottom', "time", units='ps')
    
    @pyqtSlot()
    def on_pushButton_global_get_simulated_clicked(self):
        '''
        load the simulated TRPES
        '''
        self.global_TRPES_original_for_bg=None
        self.global_TRPES=self.sim_2D_TRPES
        self.global_time=self.time_fit_range
        self.global_eV=self.pe_energy_range
        self.comboBox_global_TRPES_which.blockSignals(True)
        self.comboBox_global_TRPES_which.clear()
        self.comboBox_global_TRPES_which.addItem('Original')
        self.comboBox_global_TRPES_which.setCurrentIndex(0)
        self.comboBox_global_TRPES_which.blockSignals(False)
        self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
        self.viewer_global_pe_gauss_update()
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        self.checkBox_global_subtract_background.setChecked(False)
        self.initalize_spinboxes_new_dataset()
        self.global_TRPES_dataset=self.global_TRPES
        self.global_time_dataset=self.global_time
        self.global_eV_dataset=self.global_eV
        
    @pyqtSlot()
    def on_pushButton_get_reconstructed_TRPES_clicked(self):
        '''
        get the reconstructed TRPES from the SVD tab
        '''
        if type(self.SVD_reconstructed)!=type(None):
            #all variables global analysis
            self.global_TRPES_original_for_bg=None
            self.global_TRPES=self.SVD_reconstructed
            self.global_time=self.SVD_time
            self.global_eV=self.SVD_eV
            self.comboBox_global_TRPES_which.blockSignals(True)
            self.comboBox_global_TRPES_which.clear()
            self.comboBox_global_TRPES_which.addItem('Original')
            self.comboBox_global_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_TRPES_which.blockSignals(False)
            self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            self.viewer_global_pe_gauss_update()
            self.update_global_fit_parameters()
            self.viewer_global_time_guess_update()
            self.checkBox_global_subtract_background.setChecked(False)
            self.initalize_spinboxes_new_dataset()
            self.global_TRPES_dataset=self.global_TRPES
            self.global_time_dataset=self.global_time
            self.global_eV_dataset=self.global_eV
        
    @pyqtSlot()
    def on_pushButton_global_load_trpes_clicked(self):
        '''
        load trpes from file
        '''
        self.global_TRPES_original_for_bg=None
        filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
        self.dir=filename[:(-len(filename.split('/')[-1]))]
        if filename[-3:]=='csv':
            print('it loaded as a csv file')
            #load the data as a csv file
            self.global_time,self.global_eV,self.global_TRPES=self.load_csv(filename)
            self.comboBox_global_TRPES_which.blockSignals(True)
            self.comboBox_global_TRPES_which.clear()
            self.comboBox_global_TRPES_which.addItem('Original')
            self.comboBox_global_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_TRPES_which.blockSignals(False)
            self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            self.viewer_global_pe_gauss_update()
            self.update_global_fit_parameters()
            self.viewer_global_time_guess_update()
            self.checkBox_global_subtract_background.setChecked(False)
            self.initalize_spinboxes_new_dataset()
            self.global_TRPES_dataset=self.global_TRPES
            self.global_time_dataset=self.global_time
            self.global_eV_dataset=self.global_eV
        elif filename!='':
            #load my old data format
            loaded=np.loadtxt(filename,comments='# ')
            self.global_time,self.global_eV,self.global_TRPES=self.convert_data_to_plottable(loaded)
            self.comboBox_global_TRPES_which.blockSignals(True)
            self.comboBox_global_TRPES_which.clear()
            self.comboBox_global_TRPES_which.addItem('Original')
            self.comboBox_global_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_TRPES_which.blockSignals(False)
            self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            self.viewer_global_pe_gauss_update()
            self.update_global_fit_parameters()
            self.viewer_global_time_guess_update()
            self.checkBox_global_subtract_background.setChecked(False)
            self.initalize_spinboxes_new_dataset()
            self.global_TRPES_dataset=self.global_TRPES
            self.global_time_dataset=self.global_time
            self.global_eV_dataset=self.global_eV

    def initalize_spinboxes_new_dataset(self):
        #sets limits time for the currently loaded dataset
        min_value_time=self.global_time.min()
        max_value_time=self.global_time.max()
        min_value_eV=self.global_eV.min()
        max_value_eV=self.global_eV.max()
        self.doubleSpinBox_global_dataset_time_min.blockSignals(True)
        self.doubleSpinBox_global_dataset_time_max.blockSignals(True)
        self.doubleSpinBox_global_dataset_eV_min.blockSignals(True)
        self.doubleSpinBox_global_dataset_eV_max.blockSignals(True)
        self.doubleSpinBox_global_dataset_time_min.setValue(min_value_time)
        self.doubleSpinBox_global_dataset_time_max.setValue(max_value_time)
        self.doubleSpinBox_global_dataset_eV_min.setValue(min_value_eV)
        self.doubleSpinBox_global_dataset_eV_max.setValue(max_value_eV)
        self.doubleSpinBox_global_dataset_time_min.blockSignals(False)
        self.doubleSpinBox_global_dataset_time_max.blockSignals(False)
        self.doubleSpinBox_global_dataset_eV_min.blockSignals(False)
        self.doubleSpinBox_global_dataset_eV_max.blockSignals(False)

    @pyqtSlot()
    def on_pushButton_global_make_new_dataset_clicked(self):
        #make a new restricted dataset,but keep the original stuff
        if type(self.global_TRPES)!=type(None):
            self.comboBox_global_TRPES_which.setCurrentIndex(0)
            #making the new dataset
            eV_min=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_dataset_eV_min.value())
            eV_max=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_dataset_eV_max.value())
            if eV_min>eV_max:
                eV_min=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_dataset_eV_max.value())
                eV_max=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_dataset_eV_min.value())
            if eV_min==eV_max:
                if eV_min!=0:
                    eV_min-=1
                else:
                    eV_max+=1
            time_min=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_dataset_time_min.value())
            time_max=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_dataset_time_max.value())
            if time_min>time_max:
                time_min=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_dataset_time_max.value())
                time_max=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_dataset_time_min.value())
            if time_min==time_max:
                if time_min!=0:
                    time_min-=1
                else:
                    time_max+=1
            self.global_TRPES_dataset=copy.deepcopy(self.global_TRPES[time_min:time_max+1,eV_min:eV_max+1])
            self.global_time_dataset=copy.deepcopy(self.global_time[time_min:time_max+1])
            self.global_eV_dataset=copy.deepcopy(self.global_eV[eV_min:eV_max+1])
            #updating the combobox
            print('the y axis has the following limits',np.min(self.global_eV_dataset),np.max(self.global_eV_dataset))
            self.comboBox_global_TRPES_which.blockSignals(True)
            self.comboBox_global_TRPES_which.clear()
            self.comboBox_global_TRPES_which.addItem('Original')
            self.comboBox_global_TRPES_which.addItem('New Dataset')
            self.comboBox_global_TRPES_which.setCurrentIndex(1)
            self.on_comboBox_global_TRPES_which_currentIndexChanged(self.comboBox_global_TRPES_which.currentIndex())
            self.comboBox_global_TRPES_which.blockSignals(False)

    @pyqtSlot('int')
    def on_comboBox_global_TRPES_which_currentIndexChanged(self,value):
        if type(self.global_TRPES)!=type(None):
            if value==0:
                self.global_TRPES_original_for_bg=None
                self.checkBox_global_subtract_background.setChecked(False)
                #display original and go back!
                interm_TRPES=copy.deepcopy(self.global_TRPES)
                interm_eV=copy.deepcopy(self.global_time)
                interm_time=copy.deepcopy(self.global_eV)
                interm_TRPES=copy.deepcopy(self.global_TRPES)
                interm_eV=copy.deepcopy(self.global_eV)
                interm_time=copy.deepcopy(self.global_time)
                self.global_TRPES=copy.deepcopy(self.global_TRPES_dataset)
                self.global_time=copy.deepcopy(self.global_time_dataset)
                self.global_eV=copy.deepcopy(self.global_eV_dataset)
                self.global_TRPES_dataset=copy.deepcopy(interm_TRPES)
                self.global_time_dataset=copy.deepcopy(interm_time)
                self.global_eV_dataset=copy.deepcopy(interm_eV)

            if value==1:
                #display the new dataset
                self.global_TRPES_original_for_bg=None
                self.checkBox_global_subtract_background.setChecked(False)
                interm_TRPES=copy.deepcopy(self.global_TRPES)
                interm_eV=copy.deepcopy(self.global_eV)
                interm_time=copy.deepcopy(self.global_time)
                self.global_TRPES=copy.deepcopy(self.global_TRPES_dataset)
                self.global_time=copy.deepcopy(self.global_time_dataset)
                self.global_eV=copy.deepcopy(self.global_eV_dataset)
                self.global_TRPES_dataset=copy.deepcopy(interm_TRPES)
                self.global_time_dataset=copy.deepcopy(interm_time)
                self.global_eV_dataset=copy.deepcopy(interm_eV)
            #plotting it
            print('This is the type!',type(self.global_TRPES))
            print('these are the plotted limits',np.min(self.global_eV),np.max(self.global_eV))
            self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            self.viewer_global_time_guess_update()
            self.viewer_global_pe_gauss_update()
    
    #----------------------------------
    #all functions regrouped from the inital guess tab
    #----------------------------------
    def viewer_global_pe_gauss_update(self):
        '''
        load the current DAS and fit it intensity-wise best to the displayed spectra (just for displaying purposes) 
        '''        
        if type(self.global_TRPES)!=type(None):
            #let's first display the summed pe
            self.viewer_global_pe_guess.clear()
            #delete the old legend
            try:
                self.viewer_global_pe_legend.scene().removeItem(self.viewer_global_pe_legend)
            except Exception as e:
               pass
            self.viewer_global_pe_legend=self.viewer_global_pe_guess.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
            v_min=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_guess_comp_min.value())
            v_max=self.find_nearest_index(self.global_time,self.doubleSpinBox_global_guess_comp_max.value())
            if v_min==v_max:
                v_min-=1
            if v_min+1==v_max:
                y=self.global_TRPES[v_min:v_max,:].transpose()
                y=np.reshape(y,self.global_eV.shape)
                pe_to_be_fitted=y
                self.viewer_global_pe_guess.plot(self.global_eV,y, symbolPen='w',name='exp')
                eV_max=max(y)
            else:
                if v_min>v_max:
                    pe_to_be_fitted_2D=self.global_TRPES[v_max:v_min,:]
                else:
                    pe_to_be_fitted_2D=self.global_TRPES[v_min:v_max,:]

                pe_to_be_fitted=np.sum(pe_to_be_fitted_2D,axis=0)/np.max(pe_to_be_fitted_2D)
                self.viewer_global_pe_guess.plot(self.global_eV,pe_to_be_fitted, symbolPen='w',name='exp')
                eV_max=np.max(pe_to_be_fitted)
            #make the summed_Pe
            summed_pe=np.zeros(self.global_eV.shape)
            for gauss in self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()]:
                summed_pe+=gauss.calculate_gauss(self.global_eV)
            normalize_factor=max(summed_pe)
            summed_pe=np.nan_to_num(eV_max*summed_pe/normalize_factor)
            
            #plot the fit functions stuff
            for i, gauss in enumerate(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()]):
                if i==(self.comboBox_global_PE_current.currentIndex()):
                    self.viewer_global_pe_guess.plot(self.global_eV,eV_max*0.8*gauss.calculate_gauss(self.global_eV),
                                    pen=pg.mkPen(self.colors[i],width=3),name=str(i))
                else:
                    self.viewer_global_pe_guess.plot(self.global_eV,eV_max*0.8*gauss.calculate_gauss(self.global_eV),
                                    pen=self.colors[i],width=1,name=str(i))
            #plot the summed
            self.viewer_global_pe_guess.plot(self.global_eV,summed_pe,pen=(0,0,0))
        
    
    @pyqtSlot('int')
    def on_comboBox_global_PE_guess_DAS_which_currentIndexChanged(self,value):
        
        to_block=[self.spinBox_global_PE_number,
                  self.comboBox_global_PE_current]
        for b in to_block:
            b.blockSignals(True)
        #load the lowest ranking gaussian and display only that
        number_gaussians=len(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()])
        self.spinBox_global_PE_number.setValue(number_gaussians)
        self.comboBox_global_PE_current.clear()
        for i in range(number_gaussians):
            self.comboBox_global_PE_current.addItem(str(i+1))
        self.global_load_current_gauss()
        self.viewer_global_pe_gauss_update()
        for b in to_block:
            b.blockSignals(False)
            
    def global_load_current_gauss(self):
        to_block=[self.spinBox_global_PE_number,
                  self.comboBox_global_PE_current,
                  self.doubleSpinBox_global_PE_center,
                  self.doubleSpinBox_global_PE_width,
                  self.doubleSpinBox_global_pe_assymetry,
                  self.doubleSpinBox_global_pe_rel_intensity,
                  self.radioButton_center_fixed,
                  self.radioButton_width_fixed,
                  self.radioButton_assym_fixed,self.radioButton_rel_int_fixed]
        for b in to_block:
            b.blockSignals(True)
        index=self.comboBox_global_PE_current.currentIndex()
        self.doubleSpinBox_global_PE_center.setValue(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].mu)
        self.doubleSpinBox_global_PE_width.setValue(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].fwhm)
        self.doubleSpinBox_global_pe_assymetry.setValue(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].alpha)
        self.doubleSpinBox_global_pe_rel_intensity.setValue(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].intensity)
        self.radioButton_center_fixed.setChecked(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].mu_fitted)
        self.radioButton_width_fixed.setChecked(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].fwhm_fitted)
        self.radioButton_assym_fixed.setChecked(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].alpha_fitted)
        self.radioButton_rel_int_fixed.setChecked(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].intensity_fitted)
        for b in to_block:
            b.blockSignals(False)
        
    
    @pyqtSlot('double')
    def on_doubleSpinBox_global_guess_comp_min_valueChanged(self,value):
        self.doubleSpinBox_global_guess_comp_min.blockSignals(True)
        self.doubleSpinBox_global_guess_comp_max.blockSignals(True)
        if value>self.doubleSpinBox_global_guess_comp_max.value():
            self.doubleSpinBox_global_guess_comp_min.setValue(self.global_time[self.find_nearest_index(self.global_time,
                                                                                                       self.doubleSpinBox_global_guess_comp_max.value())-1])
        self.doubleSpinBox_global_guess_comp_min.blockSignals(False)
        self.doubleSpinBox_global_guess_comp_max.blockSignals(False)
        self.viewer_global_pe_gauss_update()
        

    @pyqtSlot('double')
    def on_doubleSpinBox_global_guess_comp_max_valueChanged(self,value):        
        self.doubleSpinBox_global_guess_comp_min.blockSignals(True)
        self.doubleSpinBox_global_guess_comp_max.blockSignals(True)
        if value<self.doubleSpinBox_global_guess_comp_min.value():
            self.doubleSpinBox_global_guess_comp_max.setValue(self.global_time[self.find_nearest_index(self.global_time,
                                                                                                       self.doubleSpinBox_global_guess_comp_min.value())+1])
        self.doubleSpinBox_global_guess_comp_min.blockSignals(False)
        self.doubleSpinBox_global_guess_comp_max.blockSignals(False)
        self.viewer_global_pe_gauss_update()

    def update_global_gauss(self):
        '''
        take all current values
        '''
        index=self.comboBox_global_PE_current.currentIndex()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].mu=self.doubleSpinBox_global_PE_center.value()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].fwhm=self.doubleSpinBox_global_PE_width.value()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].alpha=self.doubleSpinBox_global_pe_assymetry.value()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].intensity=self.doubleSpinBox_global_pe_rel_intensity.value()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].mu_fitted=self.radioButton_center_fixed.isChecked()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].fwhm_fitted=self.radioButton_width_fixed.isChecked()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].alpha_fitted=self.radioButton_assym_fixed.isChecked()
        self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][index].intensity_fitted=self.radioButton_rel_int_fixed.isChecked()        
        self.viewer_global_pe_gauss_update()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_global_PE_center_valueChanged(self,value):
        self.update_global_gauss()

    @pyqtSlot('double')
    def on_doubleSpinBox_global_PE_width_valueChanged(self,value):
        self.update_global_gauss()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_pe_assymetry_valueChanged(self,value):
        self.update_global_gauss()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_pe_rel_intensity_valueChanged(self,value):
        self.update_global_gauss()
    @pyqtSlot('bool')
    def on_radioButton_center_fixed_toggled(self,value):
        self.update_global_gauss()
    @pyqtSlot('bool')
    def on_radioButton_width_fixed_toggled(self,value):
        self.update_global_gauss()
    @pyqtSlot('bool')
    def on_radioButton_assym_fixed_toggled(self,value):
        self.update_global_gauss()
    @pyqtSlot('bool')
    def on_radioButton_rel_int_fixed_toggled(self,value):
        self.update_global_gauss()

    @pyqtSlot('int')
    def on_spinBox_global_PE_number_valueChanged(self,value):
        '''
        add or delete stuff as necessary
        '''
        #add a new one to the self.pe_gauss or delete it
        self.comboBox_global_PE_current.blockSignals(True)
        if value>len(self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()]):
            self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()].append(skewed_gauss(0.,0.2,0.,1.))            
            #and update self.comboBox_PE_current
            self.comboBox_global_PE_current.addItem(str(value))
        else:
            del self.global_pe_gauss[self.comboBox_global_PE_guess_DAS_which.currentIndex()][-1]
            #remove from combobox
            self.comboBox_global_PE_current.removeItem(self.comboBox_global_PE_current.count()-1)
        #put combobox and last index
        self.comboBox_global_PE_current.setCurrentIndex(self.comboBox_global_PE_current.count()-1)
        self.comboBox_global_PE_current.blockSignals(False)
        self.update_global_gauss()

    @pyqtSlot('int')
    def on_comboBox_global_PE_current_currentIndexChanged(self,value):
        self.global_load_current_gauss()
        self.update_global_gauss()

    def update_global_fit_parameters(self):
        """"
        getting the currently displayed fit parameters and updating the self.simulated_fit object
        """
        self.global_fit.tau1=self.doubleSpinBox_global_time_tau1_guess.value()
        self.global_fit.tau2=self.doubleSpinBox_global_time_tau2_guess.value()
        self.global_fit.tau3=self.doubleSpinBox_global_time_tau3_guess.value()
        self.global_fit.sigma_1=[1.]
        self.global_fit.sigma_2=[1.]
        self.global_fit.sigma_3=[1.]
        self.global_fit.fwhm=[self.doubleSpinBox_guess_time_irf_guess.value()]
        self.global_fit.moy=[0.]
        self.global_fit.time_offset=[self.doubleSpinBox_global_time_pos_irf_guess.value()]
        self.global_fit.sigma_offset=[0.5]
    
    @pyqtSlot('bool')
    def on_checkBox_global_time_display_all_toggled(self,value):
        self.viewer_global_time_guess_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_display_from_valueChanged(self,value):
        self.viewer_global_time_guess_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_display_to_valueChanged(self,value):
        self.viewer_global_time_guess_update()
    
    def viewer_global_time_guess_update(self):
        '''
        plot the graph of viewer_global_time_guess new with the selected fitting model, fit the intensities maybe at least rudimentairy?
        '''
        if type(self.global_TRPES)!=type(None):
            #let's first display the summed temperal evolution
            self.viewer_global_time_guess.clear()
            self.viewer_global_time_legend=self.viewer_global_time_guess.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
            if self.checkBox_global_time_display_all.isChecked()==True:
                temp_global=np.mean(self.global_TRPES,axis=1)
            else:
                #sum over the given limits
                idx1=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_time_display_from.value())
                idx2=self.find_nearest_index(self.global_eV,self.doubleSpinBox_global_time_display_to.value())
                if idx1>idx2:
                    temp_global=np.mean(self.global_TRPES[:,idx2:idx1+1],axis=1)
                elif idx2>idx1:
                    temp_global=np.mean(self.global_TRPES[:,idx1:idx2+1],axis=1)
                elif idx2==idx1:
                    temp_global=self.global_TRPES[:,idx1]
            max_for_norm=max(np.abs(temp_global))
            if max_for_norm==0.:
                max_for_norm=1
            self.viewer_global_time_guess.plot(self.global_time,temp_global/max_for_norm, symbolPen='w')
            #display a first fit
            t=self.global_time+self.doubleSpinBox_global_time_pos_irf_guess.value()
            all_curves_taus=[self.global_fit.mono_exp_decay(t,0),
                    self.global_fit.bi_exp_decay_population(t,0),
                    self.global_fit.tri_exp_decay_population(t,0)]
            names=['mono-exp','bi-exp','tri-exp']
            all_curves_final_offset=[self.global_fit.mono_exp_decay_final_state_pop(t,0),
                                 self.global_fit.bi_exp_decay_final_state_pop(t,0),
                                 self.global_fit.tri_exp_decay_final_state_pop(t,0)]
            all_curves_offset=self.global_fit.offset(t,0)
            time_curves=all_curves_taus[:self.comboBox_global_pick_model.currentIndex()+1]
            names=names[:self.comboBox_global_pick_model.currentIndex()+1]
            if self.radioButton_time_final_offset_2.isChecked()==True:
                time_curves.append(all_curves_final_offset[self.comboBox_global_pick_model.currentIndex()])
                names.append('offset_final')
            elif self.radioButton_time_offset_2.isChecked()==True:
                time_curves.append(all_curves_offset)
                names.append('offset')
            if self.comboBox_global_pick_model.currentIndex()==3:
                time_curves=[self.global_fit.fct_auto_corr(t,0)]
                names=['autocorr']
            #idea: implement automatic correct fit of intensities
            self.gobal_guess_time_curves=time_curves
            summed_decay=np.zeros(self.global_time.shape[0])
            for i,curve in enumerate(time_curves):
                if math.isnan(curve[0])!=True:
                    summed_decay+=curve
                else:
                    time_curves[i]=np.zeros(curve.shape)
            divider=max(summed_decay)
            if divider==0:
                divider=1.
            for i,curve in enumerate(time_curves):
                self.viewer_global_time_guess.plot(self.global_time,curve/divider*len(time_curves),
                                pen=pg.mkPen(self.colors[i],width=2),name=names[i]) 
            self.viewer_global_time_guess.plot(self.global_time,summed_decay/divider,
                                pen=pg.mkPen((0,0,0),width=2))
            

    
    @pyqtSlot('int')
    def on_comboBox_global_pick_model_currentIndexChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()

    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau1_guess_valueChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()

    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau2_guess_valueChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau3_guess_valueChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_guess_time_irf_guess_valueChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_pos_irf_guess_valueChanged(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        
    @pyqtSlot('bool')
    def on_radioButton_no_offset_2_toggled(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()
        
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_2_toggled(self,value):
        self.update_global_fit_parameters()
        self.viewer_global_time_guess_update()

    def construct_2D_fit_object(self):
        #make a fit_2D object and fill it with the current displayed values
        test_fit=fits_2D() 
        #make the values_to_be fitted and values_to_be_fitted_what and determine the fit function
        index_model=self.comboBox_global_pick_model.currentIndex()
        values_to_be_fitted=[]
        values_to_be_fitted_what=[]
        fixed_values=[]
        fixed_values_what=[]
        #adding offset and irf
        if self.radioButton_IRF_fixed.isChecked()==False:
            values_to_be_fitted.append(self.doubleSpinBox_guess_time_irf_guess.value())
            values_to_be_fitted_what.append('fwhm')
        else:
            fixed_values.append(self.doubleSpinBox_guess_time_irf_guess.value())
            fixed_values_what.append('fwhm')
        if self.radioButton_pos_fixed.isChecked()==False:
            values_to_be_fitted.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
            values_to_be_fitted_what.append('time_offset')
        else:
            fixed_values.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
            fixed_values_what.append('time_offset')
        if index_model>=0:
            #add all stuff for tau1
            if self.radioButton_tau1_fixed.isChecked()==False:
                values_to_be_fitted.append(self.doubleSpinBox_global_time_tau1_guess.value())
                values_to_be_fitted_what.append('tau1')
            else:
                fixed_values.append(self.doubleSpinBox_global_time_tau1_guess.value())
                fixed_values_what.append('tau1')
            test_fit.gauss1_n=len(self.global_pe_gauss[0])
            for i,g in enumerate(self.global_pe_gauss[0]):
                if g.mu_fitted==False:
                    values_to_be_fitted_what.append('gauss1'+' center;'+str(i))
                    values_to_be_fitted.append(g.mu)
                else:
                    fixed_values_what.append('gauss1'+' center;'+str(i))
                    fixed_values.append(g.mu)
                if g.fwhm_fitted==False:
                    values_to_be_fitted_what.append('gauss1'+' width;'+str(i))
                    values_to_be_fitted.append(g.fwhm)
                else:
                    fixed_values_what.append('gauss1' +' width;'+str(i))
                    fixed_values.append(g.fwhm)
                if g.alpha_fitted==False:
                    values_to_be_fitted_what.append('gauss1'+' assym;'+str(i))
                    values_to_be_fitted.append(g.alpha)
                else:
                    fixed_values_what.append('gauss1'+' assym;'+str(i))
                    fixed_values.append(g.alpha)
                if g.intensity_fitted==False:
                    values_to_be_fitted_what.append('gauss1'+' intens;'+str(i))
                    values_to_be_fitted.append(g.intensity)
                else:
                    fixed_values_what.append('gauss1' +' intens;'+str(i))
                    fixed_values.append(g.intensity)
            if index_model>=1:
                #add all stuff for tau2
                if self.radioButton_tau2_fixed.isChecked()==False:
                    values_to_be_fitted.append(self.doubleSpinBox_global_time_tau2_guess.value())
                    values_to_be_fitted_what.append('tau2')
                else:
                    fixed_values.append(self.doubleSpinBox_global_time_tau2_guess.value())
                    fixed_values_what.append('tau2')
                test_fit.gauss2_n=len(self.global_pe_gauss[1])
                for i,g in enumerate(self.global_pe_gauss[1]):
                    if g.mu_fitted==False:
                        values_to_be_fitted_what.append('gauss2'+' center;'+str(i))
                        values_to_be_fitted.append(g.mu)
                    else:
                        fixed_values_what.append('gauss2'+' center;'+str(i))
                        fixed_values.append(g.mu)
                    if g.fwhm_fitted==False:
                        values_to_be_fitted_what.append('gauss2'+' width;'+str(i))
                        values_to_be_fitted.append(g.fwhm)
                    else:
                        fixed_values_what.append('gauss2' +' width;'+str(i))
                        fixed_values.append(g.fwhm)
                    if g.alpha_fitted==False:
                        values_to_be_fitted_what.append('gauss2'+' assym;'+str(i))
                        values_to_be_fitted.append(g.alpha)
                    else:
                        fixed_values_what.append('gauss2'+' assym;'+str(i))
                        fixed_values.append(g.alpha)
                    if g.intensity_fitted==False:
                        values_to_be_fitted_what.append('gauss2'+' intens;'+str(i))
                        values_to_be_fitted.append(g.intensity)
                    else:
                        fixed_values_what.append('gauss2' +' intens;'+str(i))
                        fixed_values.append(g.intensity)
                
                if index_model==2:
                    #add all stuff for tau3
                    test_fit.gauss3_n=len(self.global_pe_gauss[2])
                    if self.radioButton_tau3_fixed.isChecked()==False:
                        values_to_be_fitted.append(self.doubleSpinBox_global_time_tau2_guess.value())
                        values_to_be_fitted_what.append('tau3')
                    else:
                        fixed_values.append(self.doubleSpinBox_global_time_tau2_guess.value())
                        fixed_values_what.append('tau3')
                    for i,g in enumerate(self.global_pe_gauss[2]):          
                        if g.mu_fitted==False:
                            values_to_be_fitted_what.append('gauss3'+' center;'+str(i))
                            values_to_be_fitted.append(g.mu)
                        else:
                            fixed_values_what.append('gauss3'+' center;'+str(i))
                            fixed_values.append(g.mu)
                        if g.fwhm_fitted==False:
                            values_to_be_fitted_what.append('gauss3'+' width;'+str(i))
                            values_to_be_fitted.append(g.fwhm)
                        else:
                            fixed_values_what.append('gauss3' +' width;'+str(i))
                            fixed_values.append(g.fwhm)
                        if g.alpha_fitted==False:
                            values_to_be_fitted_what.append('gauss3'+' assym;'+str(i))
                            values_to_be_fitted.append(g.alpha)
                        else:
                            fixed_values_what.append('gauss3'+' assym;'+str(i))
                            fixed_values.append(g.alpha)
                        if g.intensity_fitted==False:
                            values_to_be_fitted_what.append('gauss3'+' intens;'+str(i))
                            values_to_be_fitted.append(g.intensity)
                        else:
                            fixed_values_what.append('gauss3' +' intens;'+str(i))
                            fixed_values.append(g.intensity)
        #add offset stuff
        if self.radioButton_time_final_offset_2.isChecked()==True or self.radioButton_time_offset_2.isChecked()==True:
            #add the offset stuff
            test_fit.gaussO_n=len(self.global_pe_gauss[3])
            for i,g in enumerate(self.global_pe_gauss[3]):
                if g.mu_fitted==False:
                    values_to_be_fitted_what.append('gaussO'+' center;'+str(i))
                    values_to_be_fitted.append(g.mu)
                else:
                    fixed_values_what.append('gaussO'+' center;'+str(i))
                    fixed_values.append(g.mu)
                if g.fwhm_fitted==False:
                    values_to_be_fitted_what.append('gaussO'+' width;'+str(i))
                    values_to_be_fitted.append(g.fwhm)
                else:
                    fixed_values_what.append('gaussO' +' width;'+str(i))
                    fixed_values.append(g.fwhm)
                if g.alpha_fitted==False:
                    values_to_be_fitted_what.append('gaussO'+' assym;'+str(i))
                    values_to_be_fitted.append(g.alpha)
                else:
                    fixed_values_what.append('gaussO'+' assym;'+str(i))
                    fixed_values.append(g.alpha)
                if g.intensity_fitted==False:
                    values_to_be_fitted_what.append('gaussO'+' intens;'+str(i))
                    values_to_be_fitted.append(g.intensity)
                else:
                    fixed_values_what.append('gaussO' +' intens;'+str(i))
                    fixed_values.append(g.intensity)
        test_fit.init_gauss()
        test_fit.ptot_function(values_to_be_fitted,values_to_be_fitted_what)
        test_fit.ptot_function(fixed_values,fixed_values_what)
        return test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what
        
    def which_model_picked(self,test_fit):
        index_model=self.comboBox_global_pick_model.currentIndex()
        if self.radioButton_no_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay_2D,test_fit.bi_exp_decay_2D,test_fit.tri_exp_decay_2D]
            fit_function=possible_fitfunc[index_model]
        elif self.radioButton_time_final_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay_with_offset_final_2D,test_fit.bi_exp_decay_with_offset_final_2D,test_fit.tri_exp_decay_with_offset_final_2D]
            fit_function=possible_fitfunc[index_model]
        elif self.radioButton_time_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay_with_offset_2D,test_fit.bi_exp_decay_with_offset_2D,test_fit.tri_exp_decay_with_offset_2D]
            fit_function=possible_fitfunc[index_model]
        return fit_function
    
    @pyqtSlot()
    def on_pushButton_global_invert_time_axis_clicked(self):
        print('before',self.global_time)
        self.global_time=-np.flip(self.global_time,axis=0)
        self.global_time_dataset=-np.flip(self.global_time_dataset,axis=0)
        self.global_TRPES=np.flip(self.global_TRPES,axis=0)
        self.global_TRPES_dataset=np.flip(self.global_TRPES_dataset,axis=0)
        print('after',self.global_time)
        if type(self.global_TRPES_original_for_bg)!=type(None):
            self.global_TRPES_original_for_bg=np.flip(self.global_TRPES_original_for_bg,axis=0)
        self.on_comboBox_global_TRPES_which_currentIndexChanged(self.comboBox_global_TRPES_which.currentIndex())

        
    @pyqtSlot()
    def on_pushButton_global_fit_clicked(self):
        '''
        the core function for fitting goes here,
        using lmfit (seems to offer waaay more functionalities for fitting
        '''
        if type(self.global_TRPES)!=type(None):
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            #fitting it
            index_model=self.comboBox_global_pick_model.currentIndex()
            if index_model!=3:
                test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what=self.construct_2D_fit_object()
                test_fit.reporter=[]
                #decide which fit function you need to take
                fit_function=self.which_model_picked(test_fit)
                #fitting it!!
                t=self.global_time
                self.global_t_for_fit=t
                eV=self.global_eV
                self.global_eV_for_fit=eV
                TRPES=self.global_TRPES
                plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_be_fitted,
                          args=(values_to_be_fitted_what,fit_function,[[t],[eV]],[TRPES.transpose()]),full_output=True)
                #get the values out
                t2=t-test_fit.time_offset[0]
                self.global_TRPES_fitted=fit_function(np.reshape(t2,(t2.shape[0],1)),0,eV)
                self.global_TRPES_residual=TRPES-self.global_TRPES_fitted
                self.global_params_from_fit=[test_fit.extract_Deltas_plsqs(plsq,cov,values_to_be_fitted_what,fit_function,[[t],[eV]],
                                                                           [TRPES.transpose()],fixed_values,fixed_values_what),
                                             info,values_to_be_fitted_what,fixed_values_what,fixed_values]
                self.global_DAS_fitted=[]
                self.global_decays_fitted=[]
                test_fit.ptot_function(plsq,values_to_be_fitted_what)
                if index_model>=0:
                    self.global_DAS_fitted.append(test_fit.sigma_1(eV))
                    self.global_decays_fitted.append(test_fit.mono_exp_decay(t,0,eV))
                    if index_model>=1:
                        self.global_DAS_fitted.append(test_fit.sigma_2(eV))
                        self.global_decays_fitted.append(test_fit.bi_exp_decay_population(t,0,eV))
                        if index_model>=2:
                            self.global_decays_fitted.append(test_fit.tri_exp_decay_population(t,0,eV))
                            self.global_DAS_fitted.append(test_fit.sigma_3(eV))
                if self.radioButton_no_offset_2.isChecked()!=True:
                    self.global_DAS_fitted.append(test_fit.sigma_offset(eV))
                    if self.radioButton_time_final_offset_2.isChecked()==True:
                        possible_decays=[test_fit.mono_exp_decay_final_state_pop,test_fit.bi_exp_decay_final_state_pop,test_fit.tri_exp_decay_final_state_pop]
                        self.global_decays_fitted.append(possible_decays[index_model](t,0,eV))
                        possible_final_decays=[test_fit.mono_exp_decay_with_offset_final,test_fit.bi_exp_decay_with_offset_final,test_fit.tri_exp_decay_with_offset_final]
                        self.global_decays_fitted.append(possible_final_decays[index_model](t,0,eV))
                    elif self.radioButton_time_offset_2.isChecked()==True:
                        possible_decays=[test_fit.offset,test_fit.offset,test_fit.offset]
                        self.global_decays_fitted.append(possible_decays[index_model](t,0,eV))
                        possible_final_decays=[test_fit.mono_exp_decay_with_offset,test_fit.bi_exp_decay_with_offset,test_fit.tri_exp_decay_with_offset]
                        self.global_decays_fitted.append(possible_final_decays[index_model](t,0,eV))
                else:
                        possible_final_decays=[test_fit.mono_exp_decay,test_fit.bi_exp_decay,test_fit.tri_exp_decay]
                        self.global_decays_fitted.append(possible_final_decays[index_model](t,0,eV))
                #plot the stuff
                self.test_fit=test_fit
                self.plot_DAS()
                self.plot_global_decays()
                self.update_overview_results()
                self.comboBox_global_fit_or_residual.setCurrentIndex(0)
                self.on_comboBox_global_fit_or_residual_currentIndexChanged(0)
                self.update_optimization_viewer()
            else:
                print('Autocorrelation fits are only possible with the Fit2! method')
            QApplication.restoreOverrideCursor()
            
    
    @pyqtSlot('int')
    def on_comboBox_global_fit_or_residual_currentIndexChanged(self,value):
        '''
        display differently
        '''
        if type(self.global_TRPES_fitted)!=type(None):
            if value==0:  
                self.plot_TRPES(self.viewer_global_fitted,self.global_TRPES_fitted,self.global_t_for_fit,self.global_eV_for_fit)
                print('max',np.max(self.global_TRPES),'min',np.min(self.global_TRPES))
            elif value==1:
                self.plot_TRPES(self.viewer_global_fitted,self.global_TRPES_residual,self.global_t_for_fit,self.global_eV_for_fit)
                print('max',np.max(self.global_TRPES_residual),'min',np.min(self.global_TRPES_residual))

    def plot_DAS(self,normed=False):
        '''
        plotting on self.viewer_DAS the current DAS
        '''
        print('calling to plot DAS')
        self.viewer_DAS.clear()
        #delete the old legend
        self.viewer_global_das_legend=self.viewer_DAS.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        print('okay')
        names=['mono-exp','bi-exp','tri-exp'][:len(self.global_DAS_fitted)]
        if self.comboBox_global_pick_model.currentIndex()==3:
            names=['autocorr']
        if self.radioButton_time_final_offset_2.isChecked()==True or self.radioButton_time_offset_2.isChecked()==True:
            names.append('offset')
        if normed==False:
            max2=0
            for i,curve in enumerate(self.global_DAS_fitted):
                print('i',i)
                print(names)
                print('color',self.colors[i])
                self.viewer_DAS.plot(self.global_eV,curve.transpose()[:,0])
                self.viewer_DAS.plot(self.global_eV,curve.transpose()[:,0],pen=pg.mkPen(self.colors[i],width=2),name=names[i])
                if np.max(curve)>=max2:
                    max2=np.max(curve)
            self.viewer_DAS.setXRange(self.global_eV[0],self.global_eV[-1])
            self.viewer_DAS.setYRange(0,max2)
        else:
            print('jodidoo')
            
            for i,curve in enumerate(self.global_DAS_fitted):
                self.viewer_DAS.plot(self.global_eV,curve.transpose()[:,0]/np.max(curve),pen=pg.mkPen(self.colors[i],width=2),name=names[i])
            self.viewer_DAS.setXRange(self.global_eV[0],self.global_eV[-1])
            self.viewer_DAS.setYRange(0,1.)
            
    
    def normalize_viewer_plot_DAS(self,norm):
        print('this was called',self.normalize.isChecked())
        self.plot_DAS(normed=self.normalize.isChecked())
        
    def plot_global_decays(self):
        '''
        plotting the fitted decays on self.viewer_DAS_decays
        '''
        self.viewer_DAS_decays.clear()
        #delete the old legend
        """
        try:
            self.viewer_global_decays_legend.scene().removeItem(self.viewer_global_decays_legend)
        except Exception as e:
            pass
        """
        self.viewer_global_decays_legend=self.viewer_DAS_decays.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        names=['mono-exp','bi-exp','tri-exp'][:len(self.global_DAS_fitted)]
        if self.comboBox_global_pick_model.currentIndex()==3:
            names=['autocorr']
        if self.radioButton_time_final_offset_2.isChecked()==True or self.radioButton_time_offset_2.isChecked()==True:
            names.append('offset')
        total_decay=np.zeros(self.global_decays_fitted[0].shape)
        for i,curve in enumerate(self.global_decays_fitted[:-1]):
            self.viewer_DAS_decays.plot(self.global_t_for_fit,curve*np.max(self.global_DAS_fitted),pen=pg.mkPen(self.colors[i],width=2),name=names[i])
            total_decay+=curve*np.max(self.global_DAS_fitted)
        self.viewer_DAS_decays.plot(self.global_t_for_fit,total_decay,pen=pg.mkPen((0,0,0),width=2),name='total')
        
    def update_overview_results(self,second_fit=False):
        '''
        update the textBrowser_global_overview_results with the results
        '''
        text='Overview of the fitting results:\n'
        text+='_________________________________\n'
        text+='Number of iterations ' +str(self.global_params_from_fit[1]['nfev'])+'\n'
        text+='_________________________________\n'
        if second_fit==False:
            text+='DAS approximated as Gaussian spectra (Fit)\n'
        else:
            text+='No DAS assumed a priori (Fit2)\n'

        for i in range(len(self.global_params_from_fit[2])):
            unit=' ps'
            if 'gauss' in self.global_params_from_fit[2][i]:
                unit=' eV'
                if 'intensity' in self.global_params_from_fit[2][i]:
                    unit=' '
            text+=self.global_params_from_fit[2][i]+'='+'{:.4f}'.format(self.global_params_from_fit[0][0][i])+'+-'
            if type(self.global_params_from_fit[0][1][i])!=type(None) and type(self.global_params_from_fit[0][1][i])!=type('string'):
                text+='{:.4f}'.format(self.global_params_from_fit[0][1][i])+unit+'\n'
            else:
                text+=str(self.global_params_from_fit[0][1][i])+unit+'\n'
        text+='fixed values:\n'
        h=0
        for i in range(len(self.global_params_from_fit[3])):
            if 'time_offset' in self.global_params_from_fit[3][i] and h==0:
                if self.checkBox_global_floating_t0.isChecked()==True:
                    text+=self.global_params_from_fit[3][i]+'=variable\n'
                elif 'time_offset' not in text:
                    text+=self.global_params_from_fit[3][i]+'='+str(self.global_params_from_fit[4][i])+'\n'
                h=1
            if 'moy' not in self.global_params_from_fit[3][i] and 'time_offset' not in self.global_params_from_fit[3][i]:
                text+=self.global_params_from_fit[3][i]+'='+str(self.global_params_from_fit[4][i])+'\n'
        self.textBrowser_global_overview_results.setText(text)


    @pyqtSlot()
    def on_pushButton_save_global_fit_results_clicked(self):
        '''
        save the fit results if something has been fitted
        '''
        if type(self.global_TRPES_fitted)!=type(None):
            #save the stuff in four seperate files
            #.xyz the xyz file in the andrey csv format
            #.DAS the decay associated spectra in eV-y-eV-y format
            #.decay the decays in time-y-time-y-format
            #_overview.txt: the text file of the overview stuff

            #ask for filename
            filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
            #save the xyz file in andrey csv format
            data=np.zeros((self.global_TRPES_fitted.shape[0]+1,self.global_TRPES_fitted.shape[1]+1))
            data[1:,0]=self.global_t_for_fit
            data[0,1:]=self.global_eV_for_fit
            data[1:,1:]=self.global_TRPES_fitted
            np.savetxt(filename+'-xyz.csv',data,delimiter=',')
            
            data=np.zeros((self.global_TRPES.shape[0]+1,self.global_TRPES.shape[1]+1))
            data[1:,0]=self.global_t_for_fit
            data[0,1:]=self.global_eV_for_fit
            data[1:,1:]=self.global_TRPES
            np.savetxt(filename+'-original_xyz.csv',data,delimiter=',')
            
            data=np.zeros((self.global_TRPES_residual.shape[0]+1,self.global_TRPES_residual.shape[1]+1))
            data[1:,0]=self.global_t_for_fit
            data[0,1:]=self.global_eV_for_fit
            data[1:,1:]=self.global_TRPES_residual
            np.savetxt(filename+'-residual_xyz.csv',data,delimiter=',')
            #save the decay stuff
            data=np.zeros((self.global_decays_fitted[0].shape[0],len(self.global_decays_fitted)*2))
            header='all decays are saved in xy format\n'
            for i,decay in enumerate(self.global_decays_fitted):
                data[:,i*2]=self.global_t_for_fit
                data[:,i*2+1]=decay
            np.savetxt(filename+'.decay',data,header=header)
            #save the DAS stuff            
            data=np.zeros((self.global_eV_for_fit.shape[0],len(self.global_DAS_fitted)*2+1))
            header='all DAS are saved in xy format\nthe last line corresponds to the time zeros of the xyz array\n'
            for i,DAS in enumerate(self.global_DAS_fitted):
                data[:,i*2]=self.global_eV_for_fit
                data[:,i*2+1]=DAS
            data[:,-1]=self.global_t0
            np.savetxt(filename+'.DAS',data)
            #save the text overview stuff
            file=open(filename+'_overview.txt','w')
            text=self.textBrowser_global_overview_results.toPlainText()
            file.write(text)
            file.close()
            print('saved everything')

    @pyqtSlot('int')    
    def on_comboBox_global_original_vs_guess_currentIndexChanged(self,value):
        if type(self.global_TRPES)!=type(None):
            if value==0:
                self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            elif value==1:
                test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what=self.construct_2D_fit_object()
                #decide which fit function you need to take
                fit_function=self.which_model_picked(test_fit)
                t=np.reshape(self.global_time,(self.global_time.shape[0],1))
                self.plot_TRPES(self.viewer_global_orig_TRPES,fit_function(t,0,self.global_eV),self.global_time,self.global_eV)
            elif value==2:
                test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what=self.construct_2D_fit_object()
                #decide which fit function you need to take
                fit_function=self.which_model_picked(test_fit)
                t=np.reshape(self.global_time,(self.global_time.shape[0],1))
                resid=self.global_TRPES-fit_function(t,0,self.global_eV)
                self.plot_TRPES(self.viewer_global_orig_TRPES,resid,self.global_time,self.global_eV)

    def update_optimization_viewer(self):
        if self.test_fit.reporter!=[]:
            self.viewer_optimization.clear()
            self.viewer_optimization.setLogMode(x=False,y=True)
            x=range(0,len(self.test_fit.reporter))
            for i, val in enumerate(self.test_fit.reporter):
                if val>10E4:
                    del self.test_fit.reporter[i]
                    del x[i]
            self.viewer_optimization.plot(np.array(x),
                                          np.array(self.test_fit.reporter),
                                          pen=pg.mkPen((0,0,0),width=2),
                                          symbolPen='w')
            self.viewer_optimization.setLabel('left', "error", units='arb.u.')
            self.viewer_optimization.setLabel('bottom', "Number of iterations", units='')

    @pyqtSlot()                    
    def on_pushButton_global_load_DAS_clicked(self):
        '''
        load previously saved settings from a .json file
        '''
        if type(self.global_TRPES)!=type(None):
            filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir,filter="Json Files (*.json )")[0]
            #load the data of the json file
            with open(filename,'r') as read_file:
                data=json.load(read_file,object_hook=decode_anja)

            #first deactivate all signals
            self.setUpdatesEnabled(False)
            #update all values
            self.global_pe_gauss=data['self.global_pe_gauss']
            #for gauss in self.global_gauss[0]:
            #    gauss.print_gauss()
            self.comboBox_global_PE_guess_DAS_which.setCurrentIndex(data['comboBox_global_PE_guess_DAS_which'])
            self.spinBox_global_PE_number.setValue(data['spinBox_global_PE_number'])
            self.comboBox_global_PE_current.setCurrentIndex(data['comboBox_global_PE_current'])
            #time_constants:
            self.doubleSpinBox_global_time_tau1_guess.setValue(data['tau1'])
            self.radioButton_tau1_fixed.setChecked(data['tau1_fixed'])
            self.doubleSpinBox_global_time_tau1_guess.setValue(data['tau2'])
            self.radioButton_tau2_fixed.setChecked(data['tau2_fixed'])
            self.doubleSpinBox_global_time_tau1_guess.setValue(data['tau3'])
            self.radioButton_tau3_fixed.setChecked(data['tau3_fixed'])
            self.doubleSpinBox_guess_time_irf_guess.setValue(data['IRF'])
            self.radioButton_IRF_fixed.setChecked(data['IRF_fixed'])
            self.doubleSpinBox_global_time_pos_irf_guess.setValue(data['pos'])
            self.radioButton_pos_fixed.setChecked(data['pos_fixed'])
            self.radioButton_no_offset_2.setChecked(data['no_offset'])
            self.radioButton_time_final_offset_2.setChecked(data['final_offset'])
            self.radioButton_time_offset_2.setChecked(data['offset?'])
            self.comboBox_global_pick_model.setCurrentIndex(data['comboBox_global_pick_model'])

            #enable them again
            self.setUpdatesEnabled(True)
            #update all the plots
            self.viewer_global_time_guess_update()
            self.update_global_gauss()

    @pyqtSlot()
    def on_pushButton_global_save_DAS_clicked(self):
        '''
        save current settings of DAS and time stuff
        '''
        filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir,filter="Json Files (*.json )")[0]
        filename=filename.split('.')[0]
        filename+='.json'
        #creating a big dictionairy with all the values that you want to save
        data={}
        #PE
        data['self.global_pe_gauss']=self.global_pe_gauss
        data['comboBox_global_PE_guess_DAS_which']=self.comboBox_global_PE_guess_DAS_which.currentIndex()
        data['spinBox_global_PE_number']=self.spinBox_global_PE_number.value()
        data['comboBox_global_PE_current']=self.comboBox_global_PE_current.currentIndex()
        #time_constants:
        data['tau1']=self.doubleSpinBox_global_time_tau1_guess.value()
        data['tau1_fixed']=self.radioButton_tau1_fixed.isChecked()
        data['tau2']=self.doubleSpinBox_global_time_tau1_guess.value()
        data['tau2_fixed']=self.radioButton_tau2_fixed.isChecked()
        data['tau3']=self.doubleSpinBox_global_time_tau1_guess.value()
        data['tau3_fixed']=self.radioButton_tau3_fixed.isChecked()
        data['IRF']=self.doubleSpinBox_guess_time_irf_guess.value()
        data['IRF_fixed']=self.radioButton_IRF_fixed.isChecked()
        data['pos']=self.doubleSpinBox_global_time_pos_irf_guess.value()
        data['pos_fixed']=self.radioButton_pos_fixed.isChecked()
        data['no_offset']=self.radioButton_no_offset_2.isChecked()
        data['final_offset']=self.radioButton_time_final_offset_2.isChecked()
        data['offset?']=self.radioButton_time_offset_2.isChecked()
        data['comboBox_global_pick_model']=self.comboBox_global_pick_model.currentIndex()

        #dump the data
        with open(filename,'w') as write_file:
            json.dump(data,write_file,default=encode_anja)
            
    def construct_2D_object_for_fitmethod2(self,z,factors):
        '''
        z: matrix to be fitted
        factors: guess for rel. intensities of sigma_1,sigma2 etc
        returns necessary arrays for fitting
        '''
        test_fit=fits_2D_2nd()
        test_fit.reporter=[]
        index_model=self.comboBox_global_pick_model.currentIndex()
        values_to_be_fitted=[]
        values_to_be_fitted_what=[]
        fixed_values=[]
        fixed_values_what=[]
        interm_values=[]
        interm_values_what=[]
        #adding offset and irf
        if self.radioButton_IRF_fixed.isChecked()==False:
            values_to_be_fitted.append(self.doubleSpinBox_guess_time_irf_guess.value())
            values_to_be_fitted_what.append('fwhm')
        else:
            fixed_values.append(self.doubleSpinBox_guess_time_irf_guess.value())
            fixed_values_what.append('fwhm')
        if self.radioButton_pos_fixed.isChecked()==False:
            if self.checkBox_global_floating_t0.isChecked()==False:
                values_to_be_fitted.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
                values_to_be_fitted_what.append('time_offset')   
            else:
                interm_values.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
                interm_values_what.append('time_offset')
            for n in range(z.shape[1]):
                fixed_values.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
                fixed_values_what.append('time_offset')
        else:
            fixed_values.append(self.doubleSpinBox_global_time_pos_irf_guess.value())
            fixed_values_what.append('time_offset')
        if index_model==3:
            for n in range(z.shape[1]):
                interm_values.append(z[:,n].max()*factors[0])
                interm_values_what.append('sigma_1')
        elif index_model>=0:
            #add all stuff for tau1
            if self.radioButton_tau1_fixed.isChecked()==False:
                values_to_be_fitted.append(self.doubleSpinBox_global_time_tau1_guess.value())
                values_to_be_fitted_what.append('tau1')
            else:
                fixed_values.append(self.doubleSpinBox_global_time_tau1_guess.value())
                fixed_values_what.append('tau1')
            #adding the sigma values
            for n in range(z.shape[1]):
                interm_values.append(z[:,n].max()*factors[0])
                interm_values_what.append('sigma_1')
                
            if index_model>=1:
                #add all stuff for tau2
                if self.radioButton_tau2_fixed.isChecked()==False:
                    values_to_be_fitted.append(self.doubleSpinBox_global_time_tau2_guess.value())
                    values_to_be_fitted_what.append('tau2')
                else:
                    fixed_values.append(self.doubleSpinBox_global_time_tau2_guess.value())
                    fixed_values_what.append('tau2')
                #adding the sigma values
                for n in range(z.shape[1]):
                    interm_values.append(z[:,n].max()*factors[1])
                    interm_values_what.append('sigma_2')
                
                if index_model==2:
                    #add all stuff for tau3
                    test_fit.gauss3_n=len(self.global_pe_gauss[2])
                    if self.radioButton_tau3_fixed.isChecked()==False:
                        values_to_be_fitted.append(self.doubleSpinBox_global_time_tau3_guess.value())
                        values_to_be_fitted_what.append('tau3')
                    else:
                        fixed_values.append(self.doubleSpinBox_global_time_tau3_guess.value())
                        fixed_values_what.append('tau3')
                    #adding the sigma values
                    for n in range(z.shape[1]):
                        interm_values.append(z[:,n].max()*factors[2])
                        interm_values_what.append('sigma_3')

        #add offset stuff
        if self.radioButton_time_final_offset_2.isChecked()==True or self.radioButton_time_offset_2.isChecked()==True:
            for n in range(z.shape[1]):
                interm_values.append(z[:,n].max()*factors[-1])
                interm_values_what.append('sigma_offset')
        for n in range(z.shape[1]):
            fixed_values.append(np.mean(0.))
            fixed_values_what.append('moy')
            
        test_fit.ptot_function(values_to_be_fitted,values_to_be_fitted_what)
        test_fit.ptot_function(interm_values,interm_values_what)
        test_fit.ptot_function(fixed_values,fixed_values_what)
        return test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what

    def which_model_picked_for_fitmethod2(self,test_fit):
        index_model=self.comboBox_global_pick_model.currentIndex()
        if index_model==3:
            fit_func_interm=test_fit.interm_fit.fct_auto_corr
            fit_func_interm_what='fct_auto_corr'
            fit_function=test_fit.fct_auto_corr
        elif self.radioButton_no_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay,test_fit.bi_exp_decay,test_fit.tri_exp_decay]
            possible_fitfunc_interm=[test_fit.interm_fit.mono_exp_decay,test_fit.interm_fit.bi_exp_decay,test_fit.interm_fit.tri_exp_decay]
            possible_fitfunc_interm_what=['mono_exp_decay','bi_exp_decay','tri_exp_decay']
            fit_func_interm=possible_fitfunc_interm[index_model]
            fit_func_interm_what=possible_fitfunc_interm_what[index_model]
            fit_function=possible_fitfunc[index_model]
            
        elif self.radioButton_time_final_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay_with_offset_final,test_fit.bi_exp_decay_with_offset_final,test_fit.tri_exp_decay_with_offset_final]
            possible_fitfunc_interm=[test_fit.interm_fit.mono_exp_decay_with_offset_final,
                                     test_fit.interm_fit.bi_exp_decay_with_offset_final,test_fit.interm_fit.tri_exp_decay_with_offset_final]
            possible_fitfunc_interm_what=['mono_exp_decay_offset_final','bi_exp_decay_offset_final','tri_exp_decay_offset_final']
            fit_func_interm=possible_fitfunc_interm[index_model]
            fit_func_interm_what=possible_fitfunc_interm_what[index_model]
            fit_function=possible_fitfunc[index_model]
            
        elif self.radioButton_time_offset_2.isChecked()==True:
            possible_fitfunc=[test_fit.mono_exp_decay_with_offset,test_fit.bi_exp_decay_with_offset,test_fit.tri_exp_decay_with_offset]
            possible_fitfunc_interm=[test_fit.interm_fit.mono_exp_decay_with_offset,
                                     test_fit.interm_fit.bi_exp_decay_with_offset,test_fit.interm_fit.tri_exp_decay_with_offset]
            possible_fitfunc_interm_what=['mono_exp_decay_with_offset','bi_exp_decay_with_offset','tri_exp_decay_with_offset']
            fit_func_interm=possible_fitfunc_interm[index_model]
            fit_func_interm_what=possible_fitfunc_interm_what[index_model]
            fit_function=possible_fitfunc[index_model]
        return fit_function,fit_func_interm,fit_func_interm_what
           
    @pyqtSlot()
    def on_pushButton_global_fit_2_clicked(self):
        '''
        Fit the 2D Photoelectron spectra by seperatly fitting the 1D time-slices and putting the obtained sigmas together.
        '''
        if type(self.global_TRPES)!=type(None):
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            #make a new test_fit object
            #make the Ptots thingy
            index_model=self.comboBox_global_pick_model.currentIndex()
            #fitting it!!
            
            if self.actionSwitch_Csv_file_orientation.isChecked()==True:
                print('using interpolated TRPES for fitting!')
                x2=self.global_time[:-1]-self.global_time[1:]
                xi=np.arange(self.global_time.min(),self.global_time.max(),np.abs(x2).min())
                f = interpolate.interp2d(self.global_time, self.global_eV, self.global_TRPES.transpose(), kind='linear')
                TRPES=f(xi,self.global_eV).transpose()
                eV=copy.deepcopy(self.global_eV)
                self.global_eV_for_fit=eV
                self.global_t_for_fit=copy.deepcopy(xi)
                t=xi
            else:
                t=self.global_time
                self.global_t_for_fit=t
                eV=self.global_eV
                self.global_eV_for_fit=eV
                TRPES=self.global_TRPES
            #get inital guesses for the relative intensities of sigma_1,sigma_2 etc
            summed_decay=np.zeros(self.gobal_guess_time_curves[0].shape)
            
            for curve in self.gobal_guess_time_curves:
                    summed_decay+=curve
            factors=[]
            for curve in self.gobal_guess_time_curves:
                factors.append(np.max(curve/np.max(summed_decay)))
            test_fit,values_to_be_fitted,values_to_be_fitted_what,fixed_values,fixed_values_what=self.construct_2D_object_for_fitmethod2(TRPES,factors)
            #choosing the fitmodel
            fit_function,fit_func_interm,fit_func_interm_what=self.which_model_picked_for_fitmethod2(test_fit)
            #making tfit and yfit
            tfit=[]
            yfit=[]
            for n in range(TRPES.shape[1]):
                tfit.append(t)
                yfit.append(TRPES[:,n])
            #fitting it!
            plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_be_fitted,
                              args=(values_to_be_fitted_what,fit_function,tfit,yfit,
                                    fit_func_interm,fit_func_interm_what,self.checkBox_global_floating_t0.isChecked()),full_output=True,maxfev=self.spinBox_max_Iterations_fit.value())
            #sweet! Now update/get incertituted all the other stuff
            t2=t+test_fit.time_offset[0]
            test_fit.ptot_function(plsq,values_to_be_fitted_what)
            self.global_params_from_fit=[test_fit.extract_Deltas_plsqs(plsq,cov,values_to_be_fitted_what,fit_function,tfit,yfit,
                                               fit_func_interm,fit_func_interm_what),
                                         info,values_to_be_fitted_what,fixed_values_what,fixed_values]
            print(test_fit.time_offset)
            self.global_DAS_fitted=[]
            self.global_decays_fitted=[]
            if index_model==3:
                self.global_DAS_fitted.append(np.atleast_2d(np.array(test_fit.sigma_1)))
                decay=test_fit.fct_auto_corr(t2,0)
                self.global_decays_fitted.append(decay)
            elif index_model>=0:
                self.global_DAS_fitted.append(np.atleast_2d(np.array(test_fit.sigma_1)))
                test_fit.sigma_1[0]=1.
                decay=test_fit.mono_exp_decay(t2,0)
                self.global_decays_fitted.append(decay)
                if index_model>=1:
                    self.global_DAS_fitted.append(np.atleast_2d(np.array(test_fit.sigma_2)))
                    test_fit.sigma_2[0]=1.
                    decay=test_fit.bi_exp_decay_population(t2,0)
                    self.global_decays_fitted.append(decay)
                    if index_model>=2:
                        self.global_DAS_fitted.append(np.atleast_2d(np.array(test_fit.sigma_3)))
                        test_fit.sigma_3[0]=1.
                        decay=test_fit.tri_exp_decay_population(t2,0)
                        self.global_decays_fitted.append(decay)
            if self.radioButton_no_offset_2.isChecked()!=True:
                self.global_DAS_fitted.append(np.atleast_2d(np.array(test_fit.sigma_offset)))
                test_fit.sigma_offset[0]=1.
                if self.radioButton_time_final_offset_2.isChecked()==True:
                    possible_decays=[test_fit.mono_exp_decay_final_state_pop,test_fit.bi_exp_decay_final_state_pop,test_fit.tri_exp_decay_final_state_pop]
                    decay=possible_decays[index_model](t2,0)
                    self.global_decays_fitted.append(decay)
                    possible_final_decays=[test_fit.mono_exp_decay_with_offset_final,test_fit.bi_exp_decay_with_offset_final,test_fit.tri_exp_decay_with_offset_final]
                    self.global_decays_fitted.append(possible_final_decays[index_model](t2,0))
                elif self.radioButton_time_offset_2.isChecked()==True:
                    possible_decays=[test_fit.offset,test_fit.offset,test_fit.offset]
                    self.global_decays_fitted.append(possible_decays[index_model](t2,0))
                    possible_final_decays=[test_fit.mono_exp_decay_with_offset,test_fit.bi_exp_decay_with_offset,test_fit.tri_exp_decay_with_offset]
                    self.global_decays_fitted.append(possible_final_decays[index_model](t2,0))
            else:
                    possible_final_decays=[test_fit.mono_exp_decay,test_fit.bi_exp_decay,test_fit.tri_exp_decay,test_fit.fct_auto_corr]
                    self.global_decays_fitted.append(possible_final_decays[index_model](t2,0))
            #get the global_TRPES_fitted and the residual out of the DAS and possible_decay_spectra
            self.global_TRPES_fitted=np.zeros((eV.shape[0],t.shape[0]))
            if self.checkBox_global_floating_t0.isChecked()==False:
                for n,DAS in enumerate(self.global_DAS_fitted):
                    self.global_DAS_fitted[n]=np.reshape(DAS,(1,DAS.shape[1]))
                    pop=self.global_decays_fitted[n]
                    TRP_interm=np.reshape(DAS,(1,DAS.shape[1]))*np.ones((pop.shape[0],DAS.shape[1]))*np.reshape(pop,(pop.shape[0],1))
                    self.global_TRPES_fitted+=TRP_interm.transpose()
            else:
                    #make a t array:
                    times=np.array(tfit)
                    times+=np.ones(times.shape)*np.atleast_2d(test_fit.time_offset).transpose()
                    #can I put this in the intermediate function thingy?
                    test_fit.interm_fit.sigma_1=test_fit.sigma_1
                    test_fit.interm_fit.sigma_2=test_fit.sigma_2
                    test_fit.interm_fit.sigma_3=test_fit.sigma_3
                    test_fit.interm_fit.sigma_offset=test_fit.sigma_offset
                    for n in range(eV.shape[0]):
                        self.global_TRPES_fitted[n,:]=fit_func_interm(times[n,:],n)
                    self.global_TRPES_fitted[0,:]=fit_func_interm(times[0,:],0)
            self.global_t0=test_fit.time_offset            
            self.global_TRPES_fitted=self.global_TRPES_fitted.transpose()
            #self.global_TRPES_fitted=TRPES.max()*self.global_TRPES_fitted/self.global_TRPES_fitted.max()
            self.global_TRPES_residual=TRPES-self.global_TRPES_fitted 
            #plot the stuff
            self.test_fit=test_fit
            print('ev.shape',self.global_eV.shape)
            self.plot_DAS()
            self.plot_global_decays()
            self.update_overview_results(second_fit=True)
            self.comboBox_global_fit_or_residual.setCurrentIndex(0)
            self.on_comboBox_global_fit_or_residual_currentIndexChanged(0)
            self.update_optimization_viewer()
            QApplication.restoreOverrideCursor()
            
    @pyqtSlot('bool')
    def on_checkBox_global_floating_t0_toggled(self,value):
        if value==True and self.radioButton_pos_fixed.isChecked()==True:
            self.radioButton_pos_fixed.setChecked(False)
    @pyqtSlot('bool')
    def on_radioButton_pos_fixed_toggled(self,value):
        if value==True and self.checkBox_global_floating_t0.isChecked()==True:
            self.checkBox_global_floating_t0.setChecked(False)
    @pyqtSlot('double')
    def on_doubleSpinBox_global_bg_subtract_from_valueChanged(self,value):
        self.setUpdatesEnabled(False)
        if value>self.doubleSpinBox_global_bg_subtract_to.value():
            self.doubleSpinBox_global_bg_subtract_from.setValue(self.global_time[self.find_nearest_index(self.global_time,
                                                                                                         self.doubleSpinBox_global_bg_subtract_to.value())-1])
        self.setUpdatesEnabled(True)
        self.on_checkBox_global_subtract_background_toggled(self.checkBox_global_subtract_background.isChecked())

    @pyqtSlot('double')
    def on_doubleSpinBox_global_bg_subtract_to_valueChanged(self,value):
        self.setUpdatesEnabled(False)
        if value<self.doubleSpinBox_global_bg_subtract_from.value():
            self.doubleSpinBox_global_bg_subtract_to.setValue(self.global_time[self.find_nearest_index(self.global_time,
                                                                                                       self.doubleSpinBox_global_bg_subtract_from.value())+1])
        self.setUpdatesEnabled(True)
        self.on_checkBox_global_subtract_background_toggled(self.checkBox_global_subtract_background.isChecked())
            
    @pyqtSlot('bool')     
    def on_checkBox_global_subtract_background_toggled(self,toggled):
        '''
        subtract background, but keep the original file. Make therefore a transfer file.
        '''
        if type(self.global_TRPES)!=type(None):
            if type(self.global_TRPES_original_for_bg)==type(None):
                self.global_TRPES_original_for_bg=copy.deepcopy(self.global_TRPES)
            if toggled==False:
                #revert back to no background subtraction
                self.global_TRPES=copy.deepcopy(self.global_TRPES_original_for_bg)
            else:
                self.global_TRPES=self.subtract_background(self.global_TRPES_original_for_bg,self.global_time,
                                                           self.doubleSpinBox_global_bg_subtract_from.value(),
                                                           self.doubleSpinBox_global_bg_subtract_to.value())
            self.plot_TRPES(self.viewer_global_orig_TRPES,self.global_TRPES,self.global_time,self.global_eV)
            self.viewer_global_pe_gauss_update()
            self.viewer_global_time_guess_update()
            
    def subtract_background(self,z,x,value1,value2):
        index_from=self.find_nearest_index(x,value1)
        index_to=self.find_nearest_index(x,value2)
        if index_from<index_to:
            to_subtract=np.sum(np.atleast_2d(z[index_from:index_to+1,:]),axis=0)/(index_to+1-index_from)
        else:
            to_subtract=np.sum(np.atleast_2d(z[index_to:index_from+1,:]),axis=0)/(index_to+1-index_from)
        z2=z-np.atleast_2d(to_subtract)
        return z2
        

    ##########################
    #all functions regrouping from fit-bidir_2d
    ##########################
    def initalize_global_graphs_bidir(self):
        self.viewer_global_time_guess_bidir.setTitle('Total Photoelectron decay')
        self.viewer_global_time_guess_bidir.setLabel('left', "intensity", units='arb.u.')
        self.viewer_global_time_guess_bidir.setLabel('bottom', "time", units='ps')
        self.viewer_DAS_bidir.setLabel('left', "intensity", units='arb.u.')
        self.viewer_DAS_bidir.setLabel('bottom', "energy", units='eV')
        self.filterMenu3 = QtGui.QMenu("Normalize")
        self.normalize2 = QtGui.QAction("Normalize the DAS", self.filterMenu3,checkable=True,checked=False)
        self.normalize2.triggered.connect(lambda: self.normalize_viewer_plot_DAS_bidir('hello'))
        self.filterMenu3.addAction(self.normalize2)
        self.viewer_DAS_bidir.plotItem.ctrlMenu=[self.filterMenu3]
        self.viewer_DAS_decays_bidir.setLabel('left', "intensity", units='arb.u.')
        self.viewer_DAS_decays_bidir.setLabel('bottom', "time", units='ps')
        self.viewer_DAS_decays_bidir.plotItem.ctrlMenu=[]

    #----------------------------------
    #all functions regrouped from the upper buttons
    #----------------------------------
    @pyqtSlot()
    def on_pushButton_global_get_simulated_bidir_clicked(self):
        '''
        load the simulated TRPES_bidir
        '''
        self.global_TRPES_original_for_bg_bidir=None
        self.global_TRPES_bidir=copy.deepcopy(self.sim_2D_TRPES_bidir)
        self.sim_2D_TRPES_bidir=None
        self.global_time_bidir=copy.deepcopy(self.time_fit_range_bidir)
        self.global_eV_bidir=copy.deepcopy(self.pe_energy_range_bidir)
        self.comboBox_global_bidir_TRPES_which.blockSignals(True)
        self.comboBox_global_bidir_TRPES_which.clear()
        self.comboBox_global_bidir_TRPES_which.addItem('Original')
        self.comboBox_global_bidir_TRPES_which.setCurrentIndex(0)
        self.comboBox_global_bidir_TRPES_which.blockSignals(False)
        self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
        self.viewer_global_time_guess_bidir_update()
        self.checkBox_global_subtract_background_bidir.setChecked(False)
        self.initalize_spinboxes_new_dataset_bidir()
        
    @pyqtSlot()
    def on_pushButton_get_reconstructed_TRPES_bidir_clicked(self):
        '''
        get the reconstructed TRPES from the SVD tab
        '''
        if type(self.SVD_reconstructed)!=type(None):
            #all variables global analysis
            self.global_TRPES_original_for_bg_bidir=None
            self.global_TRPES_bidir=copy.deepcopy(self.SVD_reconstructed)
            self.global_time_bidir=copy.deepcopy(self.SVD_time)
            self.global_eV_bidir=copy.deepcopy(self.SVD_eV)
            self.comboBox_global_bidir_TRPES_which.blockSignals(True)
            self.comboBox_global_bidir_TRPES_which.clear()
            self.comboBox_global_bidir_TRPES_which.addItem('Original')
            self.comboBox_global_bidir_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_bidir_TRPES_which.blockSignals(False)
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
            self.viewer_global_time_guess_bidir_update()
            self.checkBox_global_subtract_background_bidir.setChecked(False)
            self.initalize_spinboxes_new_dataset_bidir()
        
    @pyqtSlot()
    def on_pushButton_global_load_trpes_bidir_clicked(self):
        '''
        load trpes from file
        '''
        self.global_TRPES_original_for_bg_bidir=None
        filename = QFileDialog.getOpenFileName(self, 'Open File',self.dir)[0]
        self.dir=filename[:(-len(filename.split('/')[-1]))]
        if filename[-3:]=='csv':
            self.comboBox_global_bidir_TRPES_which.blockSignals(True)
            self.comboBox_global_bidir_TRPES_which.clear()
            self.comboBox_global_bidir_TRPES_which.addItem('Original')
            self.comboBox_global_bidir_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_bidir_TRPES_which.blockSignals(False)
            #load the data as a csv file
            self.global_time_bidir,self.global_eV_bidir,self.global_TRPES_bidir=self.load_csv(filename)
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
            self.viewer_global_time_guess_bidir_update()
            self.checkBox_global_subtract_background_bidir.setChecked(False)
            self.initalize_spinboxes_new_dataset_bidir()
        elif filename!='':
            self.comboBox_global_bidir_TRPES_which.blockSignals(True)
            self.comboBox_global_bidir_TRPES_which.clear()
            self.comboBox_global_bidir_TRPES_which.addItem('Original')
            self.comboBox_global_bidir_TRPES_which.setCurrentIndex(0)
            self.comboBox_global_bidir_TRPES_which.blockSignals(False)
            #load my old data format
            loaded=np.loadtxt(filename,comments='# ')
            self.global_time_bidir,self.global_eV_bidir,self.global_TRPES_bidir=self.convert_data_to_plottable(loaded)
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
            self.viewer_global_time_guess_bidir_update()
            self.checkBox_global_subtract_background_bidir.setChecked(False)
            self.initalize_spinboxes_new_dataset_bidir()
            
    def initalize_spinboxes_new_dataset_bidir(self):
        #sets limits time for the currently loaded dataset
        min_value_time=self.global_time_bidir.min()
        max_value_time=self.global_time_bidir.max()
        min_value_eV=self.global_eV_bidir.min()
        max_value_eV=self.global_eV_bidir.max()
        self.doubleSpinBox_global_bidir_dataset_time_min.blockSignals(True)
        self.doubleSpinBox_global_bidir_dataset_time_max.blockSignals(True)
        self.doubleSpinBox_global_bidir_dataset_eV_min.blockSignals(True)
        self.doubleSpinBox_global_bidir_dataset_eV_max.blockSignals(True)
        self.doubleSpinBox_global_bidir_dataset_time_min.setValue(min_value_time)
        self.doubleSpinBox_global_bidir_dataset_time_max.setValue(max_value_time)
        self.doubleSpinBox_global_bidir_dataset_eV_min.setValue(min_value_eV)
        self.doubleSpinBox_global_bidir_dataset_eV_max.setValue(max_value_eV)
        self.doubleSpinBox_global_bidir_dataset_time_min.blockSignals(False)
        self.doubleSpinBox_global_bidir_dataset_time_max.blockSignals(False)
        self.doubleSpinBox_global_bidir_dataset_eV_min.blockSignals(False)
        self.doubleSpinBox_global_bidir_dataset_eV_max.blockSignals(False)

    @pyqtSlot()
    def on_pushButton_global_bidir_make_new_dataset_clicked(self):
        #make a new restricted dataset,but keep the original stuff
        if type(self.global_TRPES_bidir)!=type(None):
            self.comboBox_global_bidir_TRPES_which.setCurrentIndex(0)
            #making the new dataset
            eV_min=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_dataset_eV_min.value())
            eV_max=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_dataset_eV_max.value())
            if eV_min>eV_max:
                eV_min=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_dataset_eV_max.value())
                eV_max=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_dataset_eV_min.value())
            if eV_min==eV_max:
                if eV_min!=0:
                    eV_min-=1
                else:
                    eV_max+=1
            time_min=self.find_nearest_index(self.global_time_bidir,self.doubleSpinBox_global_bidir_dataset_time_min.value())
            time_max=self.find_nearest_index(self.global_time_bidir,self.doubleSpinBox_global_bidir_dataset_time_max.value())
            if time_min>time_max:
                time_min=self.find_nearest_index(self.global_time_bidir,self.doubleSpinBox_global_bidir_dataset_time_max.value())
                time_max=self.find_nearest_index(self.global_time_bidir,self.doubleSpinBox_global_bidir_dataset_time_min.value())
            if time_min==time_max:
                if time_min!=0:
                    time_min-=1
                else:
                    time_max+=1
            self.global_TRPES_bidir_dataset=copy.deepcopy(self.global_TRPES_bidir[time_min:time_max+1,eV_min:eV_max+1])
            self.global_time_bidir_dataset=copy.deepcopy(self.global_time_bidir[time_min:time_max+1])
            self.global_eV_bidir_dataset=copy.deepcopy(self.global_eV_bidir[eV_min:eV_max+1])
            #updating the combobox
            self.comboBox_global_bidir_TRPES_which.blockSignals(True)
            self.comboBox_global_bidir_TRPES_which.clear()
            self.comboBox_global_bidir_TRPES_which.addItem('Original')
            self.comboBox_global_bidir_TRPES_which.addItem('New Dataset')
            self.comboBox_global_bidir_TRPES_which.setCurrentIndex(1)
            self.on_comboBox_global_bidir_TRPES_which_currentIndexChanged(self.comboBox_global_bidir_TRPES_which.currentIndex())
            self.comboBox_global_bidir_TRPES_which.blockSignals(False)

    @pyqtSlot('int')
    def on_comboBox_global_bidir_TRPES_which_currentIndexChanged(self,value):
        if type(self.global_TRPES_bidir)!=type(None):
            print('called here!!')
            if value==0:
                self.checkBox_global_subtract_background_bidir.setChecked(False)
                self.global_TRPES_original_for_bg_bidir=None
                #display original and go back!
                interm_TRPES=copy.deepcopy(self.global_TRPES_bidir)
                interm_eV=copy.deepcopy(self.global_time_bidir)
                interm_time=copy.deepcopy(self.global_eV_bidir)
                interm_TRPES=copy.deepcopy(self.global_TRPES_bidir)
                interm_eV=copy.deepcopy(self.global_eV_bidir)
                interm_time=copy.deepcopy(self.global_time_bidir)
                self.global_TRPES_bidir=copy.deepcopy(self.global_TRPES_bidir_dataset)
                self.global_time_bidir=copy.deepcopy(self.global_time_bidir_dataset)
                self.global_eV_bidir=copy.deepcopy(self.global_eV_bidir_dataset)
                self.global_TRPES_bidir_dataset=copy.deepcopy(interm_TRPES)
                self.global_time_bidir_dataset=copy.deepcopy(interm_time)
                self.global_eV_bidir_dataset=copy.deepcopy(interm_eV)

            if value==1:
                #display the new dataset
                self.checkBox_global_subtract_background_bidir.setChecked(False)
                self.global_TRPES_original_for_bg_bidir=None
                interm_TRPES=copy.deepcopy(self.global_TRPES_bidir)
                interm_eV=copy.deepcopy(self.global_eV_bidir)
                interm_time=copy.deepcopy(self.global_time_bidir)
                self.global_TRPES_bidir=copy.deepcopy(self.global_TRPES_bidir_dataset)
                self.global_time_bidir=copy.deepcopy(self.global_time_bidir_dataset)
                self.global_eV_bidir=copy.deepcopy(self.global_eV_bidir_dataset)
                self.global_TRPES_bidir_dataset=copy.deepcopy(interm_TRPES)
                self.global_time_bidir_dataset=copy.deepcopy(interm_time)
                self.global_eV_bidir_dataset=copy.deepcopy(interm_eV)
            #plotting it
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
            self.viewer_global_time_guess_bidir_update()
        
    
    @pyqtSlot('double')
    def on_doubleSpinBox_global_bg_subtract_from_bidir_valueChanged(self,value):
        if type(self.global_TRPES_bidir)!=type(None):
            self.doubleSpinBox_global_bg_subtract_from_bidir.blockSignals(True)
            if value>self.doubleSpinBox_global_bg_subtract_to_bidir.value():
                self.doubleSpinBox_global_bg_subtract_from_bidir.setValue(self.global_time_bidir[self.find_nearest_index(self.global_time_bidir,
                                                                                                             self.doubleSpinBox_global_bg_subtract_to_bidir.value())-1])
            self.doubleSpinBox_global_bg_subtract_from_bidir.blockSignals(False)
            if self.checkBox_global_subtract_background_bidir.isChecked()==True:
                self.on_checkBox_global_subtract_background_bidir_toggled(self.checkBox_global_subtract_background_bidir.isChecked())

    @pyqtSlot('double')
    def on_doubleSpinBox_global_bg_subtract_to_bidir_valueChanged(self,value):
        if type(self.global_TRPES_bidir)!=type(None):
            self.doubleSpinBox_global_bg_subtract_to_bidir.blockSignals(True)
            if value<self.doubleSpinBox_global_bg_subtract_from_bidir.value():
                self.doubleSpinBox_global_bg_subtract_to_bidir.setValue(self.global_time_bidir[self.find_nearest_index(self.global_time_bidir,
                                                                                                           self.doubleSpinBox_global_bg_subtract_from_bidir.value())+1])
            self.doubleSpinBox_global_bg_subtract_to_bidir.blockSignals(False)
            if self.checkBox_global_subtract_background_bidir.isChecked()==True:
                self.on_checkBox_global_subtract_background_bidir_toggled(self.checkBox_global_subtract_background_bidir.isChecked())
            
    @pyqtSlot('bool')     
    def on_checkBox_global_subtract_background_bidir_toggled(self,toggled):
        '''
        subtract background, but keep the original file. Make therefore a transfer file.
        '''
        if type(self.global_TRPES_fitted_bidir)==type(None):
            if type(self.global_TRPES_original_for_bg_bidir)==type(None):
                self.global_TRPES_original_for_bg_bidir=copy.deepcopy(self.global_TRPES_bidir)
            if toggled==False:
                #revert back to no background subtraction
                self.global_TRPES_bidir=copy.deepcopy(self.global_TRPES_original_for_bg_bidir)
            else:
                #subtract the background
                self.global_TRPES_bidir=self.subtract_background(self.global_TRPES_original_for_bg_bidir,
                                                                 self.global_time_bidir,
                                                                 self.doubleSpinBox_global_bg_subtract_from_bidir.value(),
                                                                 self.doubleSpinBox_global_bg_subtract_to_bidir.value())
            self.plot_TRPES(self.viewer_global_orig_TRPES_bidir,self.global_TRPES_bidir,self.global_time_bidir,self.global_eV_bidir)
            self.viewer_global_time_guess_bidir_update()
            
    #----------------------------------
    #all functions regrouped from the time buttons
    #----------------------------------
    @pyqtSlot('bool')
    def on_checkBox_global_bidir_time_display_all_toggled(self,value):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_bidir_time_display_from_valueChanged(self,value):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_bidir_time_display_to_valueChanged(self,value):
        self.viewer_global_time_guess_bidir_update()
    
    def viewer_global_time_guess_bidir_update(self):
        
        if type(self.global_TRPES_bidir)!=type(None):
            #let's first display the summed temperal evolution
            self.viewer_global_time_guess_bidir.clear()
            #delete the old legend
            """
            try:
                self.viewer_global_time_legend_bidir_guess.scene().removeItem(self.viewer_global_time_legend_bidir_guess)
            except Exception as e:
                pass
            """
            self.viewer_global_time_legend_bidir_guess=self.viewer_global_time_guess_bidir.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
            if self.checkBox_global_bidir_time_display_all.isChecked()==True:
                temp_global=np.mean(self.global_TRPES_bidir,axis=1)
            else:
                #sum over the given limits
                idx1=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_time_display_from.value())
                idx2=self.find_nearest_index(self.global_eV_bidir,self.doubleSpinBox_global_bidir_time_display_to.value())
                if idx1>idx2:
                    temp_global=np.mean(self.global_TRPES_bidir[:,idx2:idx1+1],axis=1)
                elif idx2>idx1:
                    temp_global=np.mean(self.global_TRPES_bidir[:,idx1:idx2+1],axis=1)
                elif idx2==idx1:
                    temp_global=self.global_TRPES_bidir[:,idx1]
            max_for_norm=max(np.abs(temp_global))
            if max_for_norm==0.:
                max_for_norm=1
            self.global_total_pe_decay_bidir=temp_global
            self.viewer_global_time_guess_bidir.plot(self.global_time_bidir,temp_global/max_for_norm, symbolPen='w',name='exp')
            #display a first fit
            #get the stuff in the positive direction
            t=self.global_time_bidir+self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value()
            all_curves_taus=[self.global_fit_bidir.mono_exp_decay(t,0),
                    self.global_fit_bidir.bi_exp_decay_population(t,0),
                    self.global_fit_bidir.tri_exp_decay_population(t,0)]
            names=['mono-exp','bi-exp','tri-exp']
            all_curves_final_offset=[self.global_fit_bidir.mono_exp_decay_final_state_pop(t,0),
                                 self.global_fit_bidir.bi_exp_decay_final_state_pop(t,0),
                                 self.global_fit_bidir.tri_exp_decay_final_state_pop(t,0)]
            all_curves_offset=self.global_fit_bidir.offset(t,0) 
            time_curves_pos=all_curves_taus[:self.comboBox_global_pick_model_bidir_pos_3.currentIndex()+1]
            names_pos=names[:self.comboBox_global_pick_model_bidir_pos_3.currentIndex()+1]
            if self.radioButton_time_final_offset_bidir_pos_3.isChecked()==True:
                time_curves_pos.append(all_curves_final_offset[self.comboBox_global_pick_model_bidir_pos_3.currentIndex()])
                names_pos.append('offset')
            elif self.radioButton_time_offset_bidir_pos_3.isChecked()==True:
                time_curves_pos.append(all_curves_offset)
                names_pos.append('offset')
            #get the stuff in the negative direction
            all_curves_taus=[self.global_fit_bidir.mono_exp_decay(-t,1),
                    self.global_fit_bidir.bi_exp_decay_population(-t,1),
                    self.global_fit_bidir.tri_exp_decay_population(-t,1)]
            all_curves_final_offset=[self.global_fit_bidir.mono_exp_decay_final_state_pop(-t,1),
                                 self.global_fit_bidir.bi_exp_decay_final_state_pop(-t,1),
                                 self.global_fit_bidir.tri_exp_decay_final_state_pop(-t,1)]
            all_curves_offset=self.global_fit_bidir.offset(-t,1) 
            time_curves_min=all_curves_taus[:self.comboBox_global_pick_model_bidir_min_2.currentIndex()+1]
            names_min=names[:self.comboBox_global_pick_model_bidir_min_2.currentIndex()+1]
            if self.radioButton_time_final_offset_bidir_min_2.isChecked()==True:
                time_curves_min.append(all_curves_final_offset[self.comboBox_global_pick_model_bidir_min_2.currentIndex()])
                names_min.append('offset')
            elif self.radioButton_time_offset_bidir_min_2.isChecked()==True:
                time_curves_min.append(all_curves_offset)
                names_min.append('offset')
            summed_decay=np.zeros(self.global_time_bidir.shape[0])
            
            for i,curve in enumerate(time_curves_min):
                self.viewer_global_time_guess_bidir.plot(self.global_time_bidir,curve,
                                pen=pg.mkPen(self.colors[i],width=2,style=QtCore.Qt.DashLine),name=names_min[i]) 
                summed_decay+=curve
            for i,curve in enumerate(time_curves_pos):
                self.viewer_global_time_guess_bidir.plot(self.global_time_bidir,curve,
                                pen=pg.mkPen(self.colors[i],width=2,style=QtCore.Qt.SolidLine),name=names_pos[i]) 
                summed_decay+=curve
            self.viewer_global_time_guess_bidir.plot(self.global_time_bidir,summed_decay,
                                pen=pg.mkPen((0,0,0),width=2))

    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau1_guess_bidir_pos_2_valueChanged(self,value):
        self.global_fit_bidir.tau1[0]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau2_guess_bidir_pos_2_valueChanged(self,value):
        self.global_fit_bidir.tau2[0]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau3_guess_bidir_pos_2_valueChanged(self,value):
        self.global_fit_bidir.tau3[0]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_guess_time_irf_guess_bidir_pos_2_valueChanged(self,value):
        self.global_fit_bidir.fwhm=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2_valueChanged(self,value):
        self.global_fit_bidir.time_offset=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('int')
    def on_comboBox_global_pick_model_bidir_pos_3_currentIndexChanged(self,value):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('bool')
    def on_radioButton_no_offset_bidir_pos_3_toggled(self):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_bidir_pos_3_toggled(self):
        self.viewer_global_time_guess_bidir_update()

    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau1_guess_bidir_min_valueChanged(self,value):
        self.global_fit_bidir.tau1[1]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau2_guess_bidir_min_valueChanged(self,value):
        self.global_fit_bidir.tau2[1]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('double')
    def on_doubleSpinBox_global_time_tau3_guess_bidir_min_valueChanged(self,value):
        self.global_fit_bidir.tau3[1]=value
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('int')
    def on_comboBox_global_pick_model_bidir_min_2_currentIndexChanged(self,value):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('bool')
    def on_radioButton_no_offset_bidir_min_2_toggled(self):
        self.viewer_global_time_guess_bidir_update()
    @pyqtSlot('bool')
    def on_radioButton_time_final_offset_bidir_min_2_toggled(self):
        self.viewer_global_time_guess_bidir_update()

    #----------------------------------
    #all functions regrouped from the fitting stuff
    #----------------------------------
    @pyqtSlot('bool')
    def on_radioButton_pos_fixed_bidir_pos_2_toggled(self,value):
        if value==True:
            self.checkBox_global_bidir_floating_t0.setChecked(False)
    @pyqtSlot('bool')
    def on_checkBox_global_bidir_floating_t0_toggled(self,value):
        if value==True:
            self.radioButton_pos_fixed_bidir_pos_2.setChecked(False)
    
    def update_global_fit_bidir_2d_object(self):
        '''
        update the fit_bidir_2d object with the current values
        '''
        #make all the sigmas,t,y and add them
        self.global_fit_bidir_2d=fits_bidir_2d()       
        values_interm=[]
        values_interm_what=[]
        tfit=[]
        yfit=[]
        for n in range(self.global_TRPES_bidir.shape[1]):
            tfit.append(self.global_time_bidir)
            yfit.append(self.global_TRPES_bidir[:,n])
            for i in ['sigma_1_min','sigma_1_pos','sigma_2_pos','sigma_2_min','sigma_3_pos','sigma_3_min','sigma_offset_pos','sigma_offset_min']:
                values_interm.append(max(self.global_TRPES_bidir[:,n]))
                values_interm_what.append(i)
            values_interm.append(0.)
            values_interm_what.append('moy')
        self.global_fit_bidir_2d.ptot_function(values_interm,values_interm_what)
        print(len(self.global_fit_bidir_2d.sigma_offset),len(self.global_fit_bidir_2d.sigma_offset[0]),
              len(self.global_fit_bidir_2d.sigma_offset[1]))
        #making the values_to_fit stuff
        index_model_pos=self.comboBox_global_pick_model_bidir_pos_3.currentIndex()
        index_model_min=self.comboBox_global_pick_model_bidir_min_2.currentIndex()        
        
        values_to_fit=[]
        values_to_fit_what=[]
        what=['_pos','_min']
        values_fixed=[]
        values_fixed_what=[]
        for n,model in enumerate([index_model_pos,index_model_min]):
            if model>=0:
                if [self.radioButton_tau1_fixed_bidir_pos_2.isChecked(),self.radioButton_tau1_fixed_bidir_min.isChecked()][n]==False:
                    values_to_fit.append([self.doubleSpinBox_global_time_tau1_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau1_guess_bidir_min.value()][n])
                    values_to_fit_what.append('tau1'+what[n])
                else:
                    values_fixed.append([self.doubleSpinBox_global_time_tau1_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau1_guess_bidir_min.value()][n])
                    values_fixed_what.append('tau1'+what[n])
                    
            if model>=1:
                if [self.radioButton_tau2_fixed_bidir_pos_2.isChecked(),self.radioButton_tau2_fixed_bidir_min.isChecked()][n]==False:
                    values_to_fit.append([self.doubleSpinBox_global_time_tau2_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau2_guess_bidir_min.value()][n])
                    values_to_fit_what.append('tau2'+what[n])
                else:
                    values_fixed.append([self.doubleSpinBox_global_time_tau2_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau2_guess_bidir_min.value()][n])
                    values_fixed_what.append('tau2'+what[n])                    
            if model>=2:
                if [self.radioButton_tau3_fixed_bidir_pos_2.isChecked(),self.radioButton_tau3_fixed_bidir_min.isChecked()][n]==False:
                    values_to_fit.append([self.doubleSpinBox_global_time_tau3_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau3_guess_bidir_min.value()][n])
                    values_to_fit_what.append('tau3'+what[n])
                else:
                    values_fixed.append([self.doubleSpinBox_global_time_tau3_guess_bidir_pos_2.value(),
                                          self.doubleSpinBox_global_time_tau3_guess_bidir_min.value()][n])
                    values_fixed_what.append('tau3'+what[n])
        if self.radioButton_IRF_fixed_bidir_pos_2.isChecked()==False:
            values_to_fit.append(self.doubleSpinBox_guess_time_irf_guess_bidir_pos_2.value())
            values_to_fit_what.append('fwhm')
        else:
            values_fixed.append(self.doubleSpinBox_guess_time_irf_guess_bidir_pos_2.value())
            values_fixed_what.append('fwhm')
        if self.radioButton_pos_fixed_bidir_pos_2.isChecked()==False and self.checkBox_global_bidir_floating_t0.isChecked()==False:
            values_to_fit.append(self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value())
            values_to_fit_what.append('time_offset')
        elif self.radioButton_pos_fixed_bidir_pos_2.isChecked()==True:
            values_fixed.append(self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value())
            values_fixed_what.append('time_offset')
        elif self.checkBox_global_bidir_floating_t0.isChecked()==True:
            for n in range(self.global_TRPES_bidir.shape[1]):
                values_fixed.append(self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value())
                values_fixed_what.append('time_offset')
        self.global_fit_bidir_2d.ptot_function(values_fixed,values_fixed_what)   
        #deciding which positive and negative fitfunction
        fit_func_interm=[]
        for n,model in enumerate([index_model_pos,index_model_min]):
            if [self.radioButton_no_offset_bidir_pos_3.isChecked(),self.radioButton_no_offset_bidir_min_2.isChecked()][n]==True:
                possible_fitfunc_interm=[self.global_fit_bidir_2d.interm_fit.mono_exp_decay,
                                             self.global_fit_bidir_2d.interm_fit.bi_exp_decay,
                                             self.global_fit_bidir_2d.interm_fit.tri_exp_decay]
                fit_func_interm.append(possible_fitfunc_interm[model])
                
            elif [self.radioButton_time_final_offset_bidir_pos_3.isChecked(),self.radioButton_time_final_offset_bidir_min_2.isChecked()][n]==True:
                possible_fitfunc_interm=[self.global_fit_bidir_2d.interm_fit.mono_exp_decay_with_offset_final,
                                  self.global_fit_bidir_2d.interm_fit.bi_exp_decay_with_offset_final,
                                  self.global_fit_bidir_2d.interm_fit.tri_exp_decay_with_offset_final]
                fit_func_interm.append(possible_fitfunc_interm[model])
                
            elif [self.radioButton_time_offset_bidir_pos_3.isChecked(),self.radioButton_time_offset_bidir_min_2.isChecked()][n]==True:
                possible_fitfunc_interm=[self.global_fit_bidir_2d.interm_fit.mono_exp_decay_with_offset,
                                  self.global_fit_bidir_2d.interm_fit.bi_exp_decay_with_offset,
                                  self.global_fit_bidir_2d.interm_fit.tri_exp_decay_with_offset]
                fit_func_interm.append(possible_fitfunc_interm[model])
        function_pos=fit_func_interm[0]
        function_min=fit_func_interm[1]        
        return function_pos,function_min,values_to_fit_what,values_to_fit,tfit,yfit,values_fixed,values_fixed_what

    
    @pyqtSlot()
    def on_pushButton_global_fit_bidir_clicked(self):
        #fit it!
        if type(self.global_TRPES_bidir)!=type(None):
            QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            if self.actionSwitch_Csv_file_orientation.isChecked()==True:
                print('used interpolated to fit')
                x2=self.global_time_bidir[:-1]-self.global_time_bidir[1:]
                xi=np.arange(self.global_time_bidir.min(),self.global_time_bidir.max(),np.abs(x2).min())
                f = interpolate.interp2d(self.global_time_bidir, self.global_eV_bidir, self.global_TRPES_bidir.transpose(), kind='linear')
                self.global_TRPES_bidir=f(xi,self.global_eV_bidir).transpose()
                self.global_time_bidir=copy.deepcopy(xi)
            
            function_pos,function_min,values_to_fit_what,values_to_fit,tfit,yfit,values_fixed,values_fixed_what=self.update_global_fit_bidir_2d_object()
            plsq,cov,info,msg,ier=leastsq(self.global_fit_bidir_2d.fit_function,values_to_fit,
                          args=(values_to_fit_what,function_pos,function_min,tfit,yfit,self.checkBox_global_bidir_floating_t0.isChecked()),
                          full_output=True,maxfev=self.spinBox_max_Iterations_fit_bidir.value())
            #get the deltas
            plsqs,Deltas=self.global_fit_bidir_2d.extract_Deltas_plsqs(plsq,cov,values_to_fit_what,function_pos,function_min,tfit,yfit)
            self.global_fit_bidir_2d.ptot_function(plsqs,values_to_fit_what)
            #get the DAS out+plot them
            self.get_and_plot_DAS(function_pos,function_min)
            #get the decays out+plot them
            self.global_time_bidir_fitted=self.global_time_bidir
            """
            if 'time_offset' in values_to_fit_what:
                self.global_time_bidir_fitted=self.global_time_bidir+plsq[-1]
            else:
                self.global_time_bidir_fitted=self.global_time_bidir+self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value()
            """
            self.get_and_plot_decays_bidir(function_pos,function_min)
            #make the reconstructed TRPES and plot it
            self.global_TRPES_bidir_fitted=np.zeros((self.global_eV_bidir.shape[0],self.global_time_bidir.shape[0]))
            if self.checkBox_global_bidir_floating_t0.isChecked()==False:
                self.global_fit_time_offset=[]
                for n,DAS in enumerate(self.global_DAS_bidir):
                    for k, das in enumerate(DAS):
                        self.global_DAS_bidir[n][k]=np.reshape(das,(1,das.shape[0]))
                        pop=self.global_decays_bidir[n][k]
                        TRP_interm=np.reshape(das,(1,das.shape[0]))*np.ones((pop.shape[0],das.shape[0]))*np.reshape(pop,(pop.shape[0],1))
                        self.global_TRPES_bidir_fitted+=TRP_interm.transpose()
            else:
                #make a t array
                self.global_fit_time_offset=self.global_fit_bidir_2d.time_offset
                times=np.array(tfit)
                times+=np.ones(times.shape)*np.atleast_2d(self.global_fit_bidir_2d.time_offset).transpose()
                self.global_fit_bidir_2d.interm_fit.tau1=self.global_fit_bidir_2d.tau1
                self.global_fit_bidir_2d.interm_fit.tau2=self.global_fit_bidir_2d.tau2
                self.global_fit_bidir_2d.interm_fit.tau3=self.global_fit_bidir_2d.tau3
                for n in range(self.global_eV_bidir.shape[0]):
                    self.global_fit_bidir_2d.interm_fit.sigma_1=[self.global_fit_bidir_2d.sigma_1[0][n],self.global_fit_bidir_2d.sigma_1[1][n]]
                    self.global_fit_bidir_2d.interm_fit.sigma_2=[self.global_fit_bidir_2d.sigma_2[0][n],self.global_fit_bidir_2d.sigma_2[1][n]]
                    self.global_fit_bidir_2d.interm_fit.sigma_3=[self.global_fit_bidir_2d.sigma_3[0][n],self.global_fit_bidir_2d.sigma_3[1][n]]
                    self.global_fit_bidir_2d.interm_fit.sigma_offset=[self.global_fit_bidir_2d.sigma_offset[0][n],self.global_fit_bidir_2d.sigma_offset[1][n]]
                    self.global_TRPES_bidir_fitted[n,:]=function_pos(times[n,:],0)#+function_min(-times[n,:],1)
            self.global_TRPES_bidir_fitted=self.global_TRPES_bidir_fitted.transpose()
            self.global_TRPES_residual_bidir=self.global_TRPES_bidir-self.global_TRPES_bidir_fitted
            self.comboBox_global_fit_or_residual_bidir.blockSignals(True)
            self.plot_TRPES(self.viewer_global_fitted_bidir,self.global_TRPES_bidir_fitted,self.global_time_bidir_fitted,self.global_eV_bidir)
            self.comboBox_global_fit_or_residual_bidir.setCurrentIndex(0)
            self.comboBox_global_fit_or_residual_bidir.blockSignals(False)
            #make the overview and plot it
            self.update_overview_bidir(plsq,Deltas,values_to_fit_what,values_fixed,values_fixed_what,function_pos,function_min)
            #get the updated reporter stuff
            self.update_reporter_bidir()
            QApplication.restoreOverrideCursor()
            

    def update_overview_bidir(self,plsq,Deltas,values_to_fit_what,values_fixed,values_fixed_what,function_pos,function_min):
        '''
        update the textBrowser_global_overview_results_bidir with the results
        '''
        text='Overview of the fitting results of a bidirectional fit:\n'
        text+='_________________________________\n'
        text+='Number of iterations ' +str(len(self.global_fit_bidir_2d.reporter))+'\n'
        text+='Fitmodel in positive direction:'+str(function_pos)[13:-58]+'\n'
        text+='Fitmodel in negative direction:'+str(function_min)[13:-58]+'\n'
        text+='_________________________________\n'
        unit=' ps'
        for i in range(len(values_to_fit_what)):
            text+=values_to_fit_what[i]+'='+'{:.4f}'.format(plsq[i])+'+-'
            if type(Deltas[i])!=type(None) and type(Deltas[i])!=type('string'):
                text+='{:.4f}'.format(Deltas[i])+unit+'\n'
            else:
                text+=str(Deltas[i])+unit+'\n'
        text+='fixed values:\n'
        text+='_________________________________\n'
        if 'fwhm' not in values_to_fit_what:
            text+='fwhm='+'{:.4f}'.format(self.doubleSpinBox_guess_time_irf_guess_bidir_pos_2.value())+unit+'\n'
        if 'time_offset' not in values_to_fit_what and self.checkBox_global_bidir_floating_t0.isChecked()==False:
            text+='time_offset='+'{:.4f}'.format(self.doubleSpinBox_global_time_pos_irf_guess_bidir_pos_2.value())+unit+'\n'
        for i, value in enumerate(values_fixed):
            if values_fixed_what[i]!='time_offset':
                text+=values_fixed_what[i]+'='+'{:.4f}'.format(value)+unit+'\n'
    
        self.textBrowser_global_overview_results_bidir.setText(text)

    def update_reporter_bidir(self):
        if self.global_fit_bidir_2d.reporter!=[]:
            self.viewer_optimization_bidir.clear()
            self.viewer_optimization_bidir.setLogMode(x=False,y=True)
            x=range(0,len(self.global_fit_bidir_2d.reporter))
            self.viewer_optimization_bidir.plot(np.array(x),
                                          np.array(self.global_fit_bidir_2d.reporter),
                                          pen=pg.mkPen((0,0,0),width=2),
                                          symbolPen='w')
            self.viewer_optimization_bidir.setTitle('Fit convergence')
            self.viewer_optimization_bidir.setLabel('left', "error", units='arb.u.')
            self.viewer_optimization_bidir.setLabel('bottom', "Number of iterations", units='')
                    
    def get_and_plot_decays_bidir(self,function_pos,function_min):
        self.viewer_DAS_decays_bidir.clear()
        #delete the old legend
        self.viewer_decays_bidir_legend=self.viewer_DAS_decays_bidir.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        self.global_decays_bidir=[[],[]]
        index_model_pos=self.comboBox_global_pick_model_bidir_pos_3.currentIndex()
        index_model_min=self.comboBox_global_pick_model_bidir_min_2.currentIndex()
        times=[self.global_time_bidir_fitted+self.global_fit_bidir_2d.time_offset[0],-(self.global_time_bidir_fitted+self.global_fit_bidir_2d.time_offset[0])]
        names=[[],[]]
        for n,model in enumerate([index_model_pos,index_model_min]):
            possible_fitfunc=[self.global_fit_bidir_2d.mono_exp_decay(times[n],n),
                                         self.global_fit_bidir_2d.bi_exp_decay_population(times[n],n),
                                         self.global_fit_bidir_2d.tri_exp_decay_population(times[n],n)]
            self.global_decays_bidir[n]=possible_fitfunc[:model+1]
            names_t=['mono-exp','bi-exp','tri-exp']
            names[n]=names_t[:model+1]
            if [self.radioButton_time_final_offset_bidir_pos_3.isChecked(),self.radioButton_time_final_offset_bidir_min_2.isChecked()][n]==True:
                possible_fitfunc=[self.global_fit_bidir_2d.mono_exp_decay_final_state_pop(times[n],n),
                                  self.global_fit_bidir_2d.bi_exp_decay_final_state_pop(times[n],n),
                                  self.global_fit_bidir_2d.tri_exp_decay_final_state_pop(times[n],n)]
                self.global_decays_bidir[n].append(possible_fitfunc[model])
                names_t=['mono-ex-offset','bi-exp-offset','tri-exp-offset']
                names[n].append('offset')
                
            elif [self.radioButton_time_offset_bidir_pos_3.isChecked(),self.radioButton_time_offset_bidir_min_2.isChecked()][n]==True:
                possible_fitfunc=[self.global_fit_bidir_2d.offset(times[n],n),
                                  self.global_fit_bidir_2d.offset(times[n],n),
                                  self.global_fit_bidir_2d.offset(times[n],n)]
                names[n].append('offset')
                self.global_decays_bidir[n].append(possible_fitfunc[model])
        #make summed decay
        summed_decay=np.zeros(self.global_time_bidir.shape)
        styles=[QtCore.Qt.SolidLine,QtCore.Qt.DashLine]     
        for j,which in enumerate(self.global_decays_bidir):
            for i,decay in enumerate(which):
                self.viewer_DAS_decays_bidir.plot(self.global_time_bidir_fitted,
                                                  np.max(self.global_DAS_bidir[j][i])*decay,
                                                  pen=pg.mkPen(self.colors[i],width=2,style=styles[j]),name=names[j][i])
                summed_decay+=np.max(self.global_DAS_bidir[j][i])*decay
        #plotting the summe decay
        self.viewer_DAS_decays_bidir.plot(self.global_time_bidir_fitted,summed_decay,pen=pg.mkPen((0,0,0),width=2))
        
                
    def get_and_plot_DAS(self,function_pos,function_min):
        self.global_DAS_bidir=[[],[]]
        index_model_pos=self.comboBox_global_pick_model_bidir_pos_3.currentIndex()
        index_model_min=self.comboBox_global_pick_model_bidir_min_2.currentIndex()
        names=[[],[]]
        for n,model in enumerate([index_model_pos,index_model_min]):
            if model>=0:
                self.global_DAS_bidir[n].append(np.array(self.global_fit_bidir_2d.sigma_1[n]))
                names[n].append('DAS tau1')
            if model>=1:
                self.global_DAS_bidir[n].append(np.array(self.global_fit_bidir_2d.sigma_2[n]))
                names[n].append('DAS tau2')
            if model>=2:
                self.global_DAS_bidir[n].append(np.array(self.global_fit_bidir_2d.sigma_3[n]))
                names[n].append('DAS tau3')
            if [self.radioButton_time_final_offset_bidir_pos_3.isChecked(),self.radioButton_time_final_offset_bidir_min_2.isChecked()][n]==True or [self.radioButton_time_offset_bidir_pos_3.isChecked(),self.radioButton_time_offset_bidir_min_2.isChecked()][n]==True:
                self.global_DAS_bidir[n].append(np.array(self.global_fit_bidir_2d.sigma_offset[n]))
                names[n].append('DAS offset')
        self.viewer_DAS_bidir.clear()
        self.viewer_das_bidir_legend=self.viewer_DAS_bidir.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        styles=[QtCore.Qt.SolidLine,QtCore.Qt.DashLine]
        for n in [0,1]:
            for i,curve in enumerate(self.global_DAS_bidir[n]):
                self.viewer_DAS_bidir.plot(self.global_eV_bidir,curve,pen=pg.mkPen(self.colors[i],width=2,style=styles[n]),name=names[n][i])
        
    def plot_DAS_bidir(self,normalized):
        index_model_pos=self.comboBox_global_pick_model_bidir_pos_3.currentIndex()
        index_model_min=self.comboBox_global_pick_model_bidir_min_2.currentIndex()
        names=[[],[]]
        for n,model in enumerate([index_model_pos,index_model_min]):
            if model>=0:
                names[n].append('DAS tau1')
            if model>=1:
                names[n].append('DAS tau2')
            if model>=2:
                names[n].append('DAS tau3')
            if [self.radioButton_time_final_offset_bidir_pos_3.isChecked(),self.radioButton_time_final_offset_bidir_min_2.isChecked()][n]==True or [self.radioButton_time_offset_bidir_pos_3.isChecked(),self.radioButton_time_offset_bidir_min_2.isChecked()][n]==True:
                names[n].append('DAS offset')
        #plotting it
        self.viewer_DAS_bidir.clear()
        self.viewer_das_bidir_legend=self.viewer_DAS_bidir.addLegend(size=(1.,0.1),offset=(-0.4,0.4))
        styles=[QtCore.Qt.SolidLine,QtCore.Qt.DashLine]
        if normalized==False:
            for n in [0,1]:
                for i,curve in enumerate(self.global_DAS_bidir[n]):
                    self.viewer_DAS_bidir.plot(self.global_eV_bidir,np.reshape(curve,(curve.shape[1],)),pen=pg.mkPen(self.colors[i],width=2,style=styles[n]),name=names[n][i])
        else:
            for n in [0,1]:
                for i,curve in enumerate(self.global_DAS_bidir[n]):
                    self.viewer_DAS_bidir.plot(self.global_eV_bidir,np.reshape(curve,(curve.shape[1],))/np.max(curve),pen=pg.mkPen(self.colors[i],width=2,style=styles[n]),name=names[n][i])
    
    def normalize_viewer_plot_DAS_bidir(self,hello):
        self.plot_DAS_bidir(self.normalize2.isChecked())
        

    @pyqtSlot()
    def on_pushButton_save_global_fit_results_bidir_clicked(self):
        '''
        save the fit results if something has been fitted of the bidirectional fit
        '''
        if type(self.global_TRPES_bidir_fitted)!=type(None):
            #save the stuff in four seperate files
            #.xyz the xyz file in the andrey csv format
            #.DAS the decay associated spectra in eV-y-eV-y format
            #.decay the decays in time-y-time-y-format
            #_overview.txt: the text file of the overview stuff

            #ask for filename
            filename=QFileDialog.getSaveFileName(self, 'Save File',self.dir)[0]
            #save the xyz file in andrey csv format
            data=np.zeros((self.global_TRPES_bidir_fitted.shape[0]+1,self.global_TRPES_bidir_fitted.shape[1]+1))
            data[1:,0]=self.global_time_bidir_fitted
            data[0,1:]=self.global_eV_bidir
            data[1:,1:]=self.global_TRPES_bidir_fitted
            np.savetxt(filename+'-xyz.csv',data,delimiter=',')
            
            data=np.zeros((self.global_TRPES_bidir.shape[0]+1,self.global_TRPES_bidir.shape[1]+1))
            data[1:,0]=self.global_time_bidir
            data[0,1:]=self.global_eV_bidir
            data[1:,1:]=self.global_TRPES_bidir
            np.savetxt(filename+'-original_xyz.csv',data,delimiter=',')
            
            data=np.zeros((self.global_TRPES_residual_bidir.shape[0]+1,self.global_TRPES_residual_bidir.shape[1]+1))
            data[1:,0]=self.global_time_bidir
            data[0,1:]=self.global_eV_bidir
            data[1:,1:]=self.global_TRPES_residual_bidir
            np.savetxt(filename+'-residual_xyz.csv',data,delimiter=',')
            #save the decay stuff
            data=np.zeros((self.global_time_bidir_fitted.shape[0],(len(self.global_decays_bidir[0])+len(self.global_decays_bidir[1]))*2))
            times=[self.global_time_bidir_fitted,self.global_time_bidir_fitted]
            k=0
            for j,which in enumerate(self.global_decays_bidir):
                for i,decay in enumerate(which):
                    data[:,k*2]=times[j]
                    data[:,k*2+1]=decay
                    k+=1
            np.savetxt(filename+'.decay',data)
            #save the DAS stuff            
            data=np.zeros((self.global_eV_bidir.shape[0],(len(self.global_DAS_bidir[0])+len(self.global_DAS_bidir[1]))*2))
            k=0
            print('len stuff in save',len(self.global_DAS_bidir[0]),len(self.global_DAS_bidir[0]))
            for n in [0,1]:
                for i,curve in enumerate(self.global_DAS_bidir[n]):
                    data[:,k*2]=self.global_eV_bidir
                    data[:,k*2+1]=curve
                    k+=1
            np.savetxt(filename+'.DAS',data)
            #save the text overview stuff
            file=open(filename+'_overview.txt','w')
            text=self.textBrowser_global_overview_results_bidir.toPlainText()
            file.write(text)
            file.close()
            if self.global_fit_time_offset!=[]:
                np.savetxt(filename+'_time_offset.txt',np.array(self.global_fit_time_offset))
            print('saved everything')


    @pyqtSlot('int')
    def on_comboBox_global_fit_or_residual_bidir_currentIndexChanged(self,value):
        '''
        display differently
        '''
        if type(self.global_TRPES_bidir_fitted)!=type(None):
            if value==0:  
                self.plot_TRPES(self.viewer_global_fitted_bidir,self.global_TRPES_bidir_fitted,self.global_time_bidir_fitted,self.global_eV_bidir)
                
            elif value==1:
                self.plot_TRPES(self.viewer_global_fitted_bidir,self.global_TRPES_residual_bidir,self.global_time_bidir_fitted,self.global_eV_bidir)

"""
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TRPES_simulator()
    w.show()
    sys.exit(app.exec_())
"""
if __name__ == '__main__':
    from PyQt5 import QtWidgets
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance() 
    main = TRPES_simulator()
    main.show()
    sys.exit(app.exec_())
