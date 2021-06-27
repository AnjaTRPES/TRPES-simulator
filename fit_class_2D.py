import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from skewed_gauss import skewed_gauss
#testing an offset function I derived
#all times in fs

class fits_2D(object):
    fwhm=[0.08]
    time_offset=[0]
    sigma_offset=[1]
    tau1=1.20
    tau2=2.400
    tau3=3.600
    gauss1=[]
    gauss1_n=0
    gauss2_n=0
    gauss3_n=0
    gaussO_n=0
    gauss2=[]
    gauss3=[]
    gaussO=[]
    moy=[]
    counter=0
    reporter=[]

    def sigma_1(self,eV):
        summed_pe=np.zeros(eV.shape)
        for gauss in self.gauss1:
            summed_pe+=gauss.calculate_gauss(eV)
        return np.reshape(summed_pe,(1,summed_pe.shape[0]))

    def sigma_2(self,eV):
        summed_pe=np.zeros(eV.shape)
        for gauss in self.gauss2:
            summed_pe+=gauss.calculate_gauss(eV)
        return np.reshape(summed_pe,(1,summed_pe.shape[0]))

    def sigma_3(self,eV):
        summed_pe=np.zeros(eV.shape)
        for gauss in self.gauss2:
            summed_pe+=gauss.calculate_gauss(eV)
        return np.reshape(summed_pe,(1,summed_pe.shape[0]))

    def sigma_offset(self):
        summed_pe=np.zeros(eV.shape)
        for gauss in self.gaussO:
            summed_pe+=gauss.calculate_gauss(eV)
        return np.reshape(summed_pe,(1,summed_pe.shape[0]))
        

    def __init__(self,test_param=20):
        self.test_param=test_param
    '''    
    def __getattr__(self, name):
       return self.data[name]
    '''
    def fwhm_to_sigma(self,fwhm):
        return fwhm/(2*np.sqrt(2*np.log(2)))

    def half_time_to_lambda(self,half_time):
        return 1./half_time
    
    def offset(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
    
    def offset_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_offset(eV)*self.offset(t,n,eV,several_fwhms=False)
        

    def special_erf(self,t,t1,FWHM):
        sigma=self.fwhm_to_sigma(FWHM)
        lam=self.half_time_to_lambda(t1)
        try:
            g=np.nan_to_num(np.exp(0.5*(lam*sigma)**2-lam*t)*sp.special.erfc((-t+lam*sigma**2)/(np.sqrt(2)*sigma)))
        except FloatingPointError:
            #print('encountered floating point error,fit_class_2D')
            g=np.zeros(t.shape)
        return g
    
    def mono_exp_decay(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.special_erf(t,self.tau1,fwhm)
    
    def mono_exp_decay_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_1(eV)*self.mono_exp_decay(t,n,eV,several_fwhms=False)

    def mono_exp_decay_final_state_pop(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return ((sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm))))-self.mono_exp_decay(t,n,eV))
    
    def mono_exp_decay_final_state_pop_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_offset(eV)*self.mono_exp_decay_final_state_pop(t,n,eV,several_fwhms=False)
    
    def mono_exp_decay_with_offset(self,t,n,eV,several_fwhms=False):
        return self.mono_exp_decay(t,n,eV,several_fwhms)+self.offset(t,n)
    
    def mono_exp_decay_with_offset_2D(self,t,n,eV,several_fwhms=False):
        return self.mono_exp_decay_2D(t,n,eV,several_fwhms)+self.offset_2D(t,n)
    
    def mono_exp_decay_with_offset_final(self,t,n,several_fwhms=False):
        return self.mono_exp_decay(t,n,eV,several_fwhms)+self.mono_exp_decay_final_state_pop(t,n,eV,several_fwhms)
    
    def mono_exp_decay_with_offset_final_2D(self,t,n,several_fwhms=False):
        return self.mono_exp_decay_2D(t,n,eV,several_fwhms)+self.mono_exp_decay_final_state_pop_2D(t,n,eV,several_fwhms)
    
    def bi_exp_decay_population(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return (lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_population_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_2(eV)*self.bi_exp_decay_population(t,n,eV,several_fwhms=False)
   
    def bi_exp_decay(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return self.mono_exp_decay(t,n,eV,several_fwhms)+(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_2D(self,t,n,eV,several_fwhms=False):
        return self.mono_exp_decay_2D(t,n,eV,several_fwhms)+self.sigma_2(eV)*self.bi_exp_decay_population(t,n,eV,several_fwhms)


    def bi_exp_decay_final_state_pop(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,eV,several_fwhms)
        c=self.bi_exp_decay_population(t,n,eV,several_fwhms)
        return (a-b-c)
    
    def bi_exp_decay_final_state_pop_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_offset(eV)*self.bi_exp_decay_final_state_pop(t,n,eV,several_fwhms=False)
    
    def bi_exp_decay_with_offset(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return self.mono_exp_decay_with_offset(t,n,eV,several_fwhms)+(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_with_offset_2D(self,t,n,eV,several_fwhms=False):
        return self.mono_exp_decay_with_offset_2D(t,n,eV,several_fwhms)+self.sigma_2(eV)*self.bi_exp_decay_population(t,n,eV,several_fwhms)

    def bi_exp_decay_with_offset_final(self,t,n,eV,several_fwhms=False):
        return self.bi_exp_decay(t,n,eV,several_fwhms)+self.bi_exp_decay_final_state_pop(t,n,eV,several_fwhms)

    def bi_exp_decay_with_offset_final_2D(self,t,n,eV,several_fwhms=False):
        return self.bi_exp_decay_2D(t,n,eV,several_fwhms)+self.bi_exp_decay_final_state_pop_2D(t,n,eV,several_fwhms)
    
    def tri_exp_decay_population(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        lam3=self.half_time_to_lambda(self.tau3)
        b1=(self.special_erf(t,self.tau1,fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2,fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3,fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        return lam1*lam2*(b1+b3)

    def tri_exp_decay_population_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_3(eV)*self.tri_exp_decay_population(t,n,eV,several_fwhms=False)
    
    def tri_exp_decay(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]        
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        lam3=self.half_time_to_lambda(self.tau3)
        a=self.bi_exp_decay(t,n,eV,several_fwhms)
        b1=(self.special_erf(t,self.tau1,fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2,fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3,fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return a+b
    
    def tri_exp_decay_2D(self,t,n,eV,several_fwhms=False):
        return self.bi_exp_decay_2D(t,n,eV,several_fwhms)+self.tri_exp_decay_population_2D(t,n,eV,several_fwhms=False)
    
    def tri_exp_decay_final_state_pop(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,eV,several_fwhms)
        c=self.bi_exp_decay_population(t,n,eV,several_fwhms)
        d=self.tri_exp_decay_population(t,n,eV,several_fwhms)
        return a-b-c-d

    def tri_exp_decay_final_state_pop_2D(self,t,n,eV,several_fwhms=False):
        return self.sigma_offset(eV)*self.tri_exp_decay_final_state_pop(t,n,eV,several_fwhms=False)

    def tri_exp_decay_with_offset(self,t,n,eV,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        lam3=self.half_time_to_lambda(self.tau3)
        a=self.bi_exp_decay_with_offset(t,n,eV,several_fwhms)
        b1=(self.special_erf(t,self.tau1,fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2,fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3,fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return b

    def tri_exp_decay_with_offset_2D(self,t,n,eV,several_fwhms=False):
        return self.bi_exp_decay_with_offset_2D(t,n,eV,several_fwhms)+self.sigma_3(eV)*self.tri_exp_decay_with_offset(t,n,eV,several_fwhms=False)
    
    def tri_exp_decay_with_offset_final(self,t,n,eV,several_fwhms=False):
        return self.tri_exp_decay(t,n,eV,several_fwhms)+self.tri_exp_decay_final_state_pop(t,n,eV,several_fwhms)
    
    def tri_exp_decay_with_offset_final_2D(self,t,n,eV,several_fwhms=False):
        return self.tri_exp_decay_2D(t,n,eV,several_fwhms)+self.tri_exp_decay_final_state_pop_2D(t,n,eV,several_fwhms)
    
    def fit_function(self,Ptot,what_Ptots,function,x_fit,Yfit,several_fwhms=False):
        '''
        Fit function (leastsq fit) with which the exp. values are fitted.
        Ptot: values to fit
        what_Ptots: In string form, to attribute it to the class variables
        function: Fitfunction to use
        x_fit: list of [[times],[energy]]
        Yfit: 2D array of eV*time
        '''
        #print 'fitting %i functions',len(tfit)
        self.ptot_function(Ptot,what_Ptots)
        tfit,energy_fit=x_fit[0],x_fit[1]
        return_roots=[]
        for variable in [self.time_offset,self.sigma_offset,self.sigma_1,self.sigma_2,self.sigma_3,tfit,Yfit,self.fwhm]:
            self.check_whether_list(variable)
        for n in range(len(tfit)): 
            #print('called this function n times',n)
            t=tfit[0]-self.time_offset[n]
            #print('shape t',t.shape)
            t=np.reshape(t,(t.shape[0],1))
            #print('calculating a')
            a=function(t,n,energy_fit[0],several_fwhms).transpose()
            #a=np.reshape(a,(a.shape[0]*a.shape[1],1))
            '''
            import matplotlib.pyplot as plt
            plt.figure()
            print('shapes for plotting',t.shape,energy_fit[0].shape,a.shape)
            plt.pcolormesh(t[:,0],energy_fit[0],a)
            plt.show()
            print(a)
            '''
            #print('finished calculating a')
            b=Yfit[n]
            #b=np.reshape(b,(b.shape[0]*b.shape[1],1))
            #print('shapes',a.shape,b.shape)
            #print('array of floats? a')
            '''
            for value in a:
                if type(value)!=type(3.4):
                    print(value)
            '''
            c=np.sqrt(np.power(a-b,2))
            c=np.reshape(c,(c.shape[0]*c.shape[1],1))
            #print(self.bi_ex.4_decay_2D(t[0],n,energy_fit[0],several_fwhms).shape)
            return_roots.append(c)
            #moyenne needs to be added into this somehow!
            #return_roots.append(np.sqrt(np.power(function(t,n,several_fwhms)+self.moy[n]-Yfit[n],2)))
            self.reporter.append(np.sum(return_roots))
        if len(return_roots)==1:
            #print('return roots shape',return_roots[0].shape)
            return return_roots[0][:,0]
        else:
            start=return_roots[0]
            for n in range(1,len(return_roots)):
                start=np.concatenate((start,return_roots[n]))            
            return start
    def init_gauss(self):
        self.gauss1=[]
        for n in range(self.gauss1_n):
            self.gauss1.append(skewed_gauss(0,0.2,0.,1.))
            
        self.gauss2=[]
        for n in range(self.gauss2_n):
            self.gauss2.append(skewed_gauss(0.,0.2,0.,1.))
            
        self.gauss3=[]                                   
        for n in range(self.gauss3_n):
            self.gauss3.append(skewed_gauss(0.,0.2,0.,1.))
            
        self.gaussO=[]
        for n in range(self.gaussO_n):
            self.gaussO.append(skewed_gauss(0.,0.2,0.,1.))
    
    def ptot_function(self,Ptot,whatPtots):
        '''
        with whatPtots the values in Ptot will be attributed to the values in the class
        whatPtots: list of stringex: [time_offset,fwhm,...]
        for the energy values it has to be in the order of [center,width, assym,intens]
        '''
        #print('Ptot',Ptot)
        #print('whatPtots',whatPtots)
        if 'time_offset' in whatPtots:
            self.time_offset=[]
        if 'sigma_offset' in whatPtots:
            self.sigma_offset=[]
        if 'sigma_1' in whatPtots:
            self.sigma_1=[]
        if 'sigma_2' in whatPtots:
            self.sigma_2=[]
        if 'sigma_3' in whatPtots:
            self.sigma_3=[]
        if 'moy' in whatPtots:
            self.moy=[]
        if 'fwhm' in whatPtots:
            self.fwhm=[]
        if len(Ptot)!=len(whatPtots):
            print ('len(Ptot)!=len(whatPtots)')
            #raise WhatPtotsError Need to look up how to define my own errors
        for n in range(len(whatPtots)):
            if 'gauss1' in whatPtots[n]:
                index=int(whatPtots[n].split(';')[1])
                if 'center' in whatPtots[n]:
                    self.gauss1[index].mu=np.abs(Ptot[n])
                if 'width' in whatPtots[n]:
                    self.gauss1[index].fwhm=Ptot[n]                            
                if 'assym' in whatPtots[n]:
                    self.gauss1[index].alpha=Ptot[n]
                if 'intens' in whatPtots[n]:
                    self.gauss1[index].intensity=Ptot[n] 
            if 'gauss2' in whatPtots[n]:
                index=int(whatPtots[n].split(';')[1])
                if 'center' in whatPtots[n]:
                    self.gauss2[index].mu=np.abs(Ptot[n])
                if 'width' in whatPtots[n]:
                    self.gauss2[index].fwhm=Ptot[n]                            
                if 'assym' in whatPtots[n]:
                    self.gauss2[index].alpha=Ptot[n]
                if 'intens' in whatPtots[n]:
                    self.gauss2[index].intensity=Ptot[n] 
            if 'gauss3' in whatPtots[n]:
                index=int(whatPtots[n].split(';')[1])
                if 'center' in whatPtots[n]:
                    self.gauss3[index]=np.abs(Ptot[n])
                if 'width' in whatPtots[n]:
                    self.gauss3[index].fwhm=Ptot[n]                            
                if 'assym' in whatPtots[n]:
                    self.gauss3[index].alpha=Ptot[n]
                if 'intens' in whatPtots[n]:
                    self.gauss3[index].intensity=Ptot[n]
            if 'gaussO' in whatPtots[n]:
                index=int(whatPtots[n].split(';')[1])
                if 'center' in whatPtots[n]:
                    self.gaussO[index]=np.abs(Ptot[n])
                if 'width' in whatPtots[n]:
                    self.gaussO[index].fwhm=Ptot[n]                            
                if 'assym' in whatPtots[n]:
                    self.gaussO[index].alpha=Ptot[n]
                if 'intens' in whatPtots[n]:
                    self.gaussO[index].intensity=Ptot[n] 
            if whatPtots[n]=='fwhm':
                if self.fwhm==[]:
                    self.fwhm=[Ptot[n]]
                else:
                    self.fwhm.append(Ptot[n])
            if whatPtots[n]=='time_offset':
                #print self.time_offset
                if self.time_offset==[]:
                    self.time_offset=[Ptot[n]]
                else:
                    self.time_offset.append(Ptot[n])                        
            if whatPtots[n]=='sigma_offset':
                if self.sigma_offset==[]:
                    self.sigma_offset=[Ptot[n]]
                else:
                    self.sigma_offset.append(Ptot[n])
            if whatPtots[n]=='tau1':
                self.tau1=abs(Ptot[n])
            if whatPtots[n]=='tau2':
                self.tau2=abs(Ptot[n])
            if whatPtots[n]=='tau3':
                self.tau3=abs(Ptot[n])
            if whatPtots[n]=='sigma_1':
                #print 'run', self.counter
                self.counter=self.counter+1
                if self.sigma_1==[]:
                    #print ' I set sigma_1 to an empty array'
                    self.sigma_1=[Ptot[n]]
                else:
                    self.sigma_1.append(Ptot[n])
            if whatPtots[n]=='sigma_2':
                if self.sigma_2==[]:
                    self.sigma_2=[Ptot[n]]
                else:
                    self.sigma_2.append(Ptot[n])
            if whatPtots[n]=='sigma_3':
                if self.sigma_3==[]:
                    self.sigma_3=[Ptot[n]]
                else:
                    self.sigma_3.append(Ptot[n])
            if whatPtots[n]=='moy':
                if self.moy==[]:
                    self.moy=[Ptot[n]]
                else:
                    self.moy.append(Ptot[n])


                
    def check_whether_list(self, var):
        pass
        '''
        if str(type(var))!="<type 'list'>":
            #raise an NotListError
            #and learn how to print the name of the variable
            print ('the variable should be a list! Value of the variable is',var)
        '''
    def fct_auto_corr(self,t,n,several_fwhms=False):
        sigma=self.fwhm_to_sigma(self.fwhm[n])
        return self.sigma_1[n]*1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-self.time_offset)**2)/(2*(sigma)**2))
    def fct_auto_corr_with_offset(self,t,n,several_fwhms=False):
        sigma=self.fwhm_to_sigma(self.fwhm[n])
        return self.sigma_1[n]*1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t-self.time_offset)**2)/(2*(sigma)**2))+self.offset(t,n)
     
    def extract_Deltas_plsqs(self,plsq,cov,whatPtots,function,tfit,Yfit,fixed,fixed_what):
        '''
        This subfunctions serves as to extract the incertitudes
        out of the optimized functions
        plsq: list
        whatPtots: list of the attributes of plsq
        cov: covariance, None if fit not converged
        returns: plsqs,Deltas
        '''
        if len(plsq)!=len(whatPtots):
            print ('the lengths of plsq and WhatPtots should be the same!!')
        #get residual
        Yres=self.fit_function(plsq,whatPtots,function,tfit,Yfit)
        res=np.sqrt(np.power(Yres,2).sum()/Yres.size)
        plsqs=[]
        Deltas=[]
        for n in range(len(plsq)):
            plsqs.append(plsq[n])
            if type(cov)==type(None):
                Delta='Inf'
            elif cov.any()==None:
                Delta='Inf'
            else:
                Delta=(np.sqrt(cov.diagonal())*res)[n]
            Deltas.append(Delta)
        '''
        for n,value in enumerate(fixed):
            plsqs.append(value)
            Deltas.append('fixed')
        '''
        return plsqs,Deltas
    def get_values(self, strings,n,several_fwhms):
        what_to_get=[]
        for string in strings:
            if string=='fwhm':
                if several_fwhms==True:
                    what_to_get.append(self.fwhm[n])
                else:
                    what_to_get.append(self.fwhm[0])
            if string=='time_offset':
                what_to_get.append(self.time_offset[n])       
            if string=='sigma_offset':
                what_to_get.append(self.sigma_offset[n])
            if string=='tau1':
                what_to_get.append(self.tau1)
            if string=='tau2':
                what_to_get.append(self.tau2)
            if string=='tau3':
                what_to_get.append(self.tau3)
            if string=='sigma_1':
                what_to_get.append(self.sigma_1[n])
            if string=='sigma_2':
               what_to_get.append(self.sigma_2[n])
            if string=='sigma_3':
               what_to_get.append(self.sigma_3[n])
            if string=='moy':
                what_to_get.append(self.moy[n])
        return what_to_get


            
        
def compare_fit_fixed_constants(fixed_constants,whats_fixed,values_to_be_fitted,values_to_be_fitted_what):
    to_be_popped=[]
    to_be_popped_what=[]
    for n in range(len(values_to_be_fitted_what)):
        #print 'values_to_be_fitted_what'
        if values_to_be_fitted_what[n] in whats_fixed:
            to_be_popped.append(values_to_be_fitted[n])
            to_be_popped_what.append(values_to_be_fitted_what[n])
    for value in to_be_popped:
        values_to_be_fitted.remove(value)
    for value in to_be_popped_what:
        values_to_be_fitted_what.remove(value)
    return values_to_be_fitted,values_to_be_fitted_what


if __name__=='__main__':
    print('testing this class')
    eV=np.arange(0,5,0.01)
    t=np.arange(-2,15,0.05)
    #simulating one 2D-TRPES
    simulated=fits_2D()
    gauss1=[skewed_gauss(0,0.2,0.,1.),
            skewed_gauss(1,0.5,0.,0.5)]
    gauss2=[skewed_gauss(2,0.2,0.,1.),
            skewed_gauss(0.5,0.5,0.,0.5)]
    simulated.gauss1=gauss1
    simulated.gauss2=gauss2
    simulated.tau1=0.8
    simulated.tau2=5.
    TRPES=simulated.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False)
    TRPES=TRPES.transpose()
    import matplotlib.pyplot as plt
    '''
    plt.figure()
    #TRPES=np.reshape(t,(t.shape[0],1))*np.reshape(eV,(1,eV.shape[0]))
    plt.pcolormesh(t,eV,TRPES,cmap='nipy_spectral')
    plt.show()
    '''
    #defining starting values, plotting them compared to the simulated spectrum
    test_fit=fits_2D()
    fixed_constants=[0.0]
    whats_fixed=['moy']
    test_fit.moy=[0.0]
    '''
    gauss_guess1=[skewed_gauss(0.1,0.2,0.,1.),
            skewed_gauss(0.9,0.5,0.,0.5)]
    gauss_guess2=[skewed_gauss(2,0.2,0.,1.),
            skewed_gauss(0.5,0.5,0.,0.5)]
    '''
    gauss_guess1=[skewed_gauss(0,0.2,0.,1.),
            skewed_gauss(1,0.5,0.,0.)]
    gauss_guess2=[skewed_gauss(2,0.2,0.,1.),
            skewed_gauss(0.5,0.5,0.,1.)]
    
    tau1=0.80
    tau2=4.900
    print('initial guess')
    #test_fit.gauss1=gauss_guess1
    #test_fit.gauss2=gauss_guess2
    test_fit.tau1=tau1
    test_fit.tau2=tau2
        
    values_to_be_fitted=[tau1,tau2]
    values_to_be_fitted_what=['tau1','tau2']
    test_fit.gauss1_n=len(gauss_guess1)
    for i,g in enumerate(gauss_guess1):
        values_to_be_fitted_what.append('gauss1'+' center;'+str(i))
        values_to_be_fitted.append(g.mu)
        values_to_be_fitted_what.append('gauss1'+' width;'+str(i))
        values_to_be_fitted.append(g.fwhm)
        #values_to_be_fitted_what.append('gauss1'+' assym;'+str(i))
        #values_to_be_fitted.append(g.alpha)
        values_to_be_fitted_what.append('gauss1'+' intens;'+str(i))
        values_to_be_fitted.append(g.intensity)
    test_fit.gauss2_n=len(gauss_guess2)
    for i,g in enumerate(gauss_guess2):
        values_to_be_fitted_what.append('gauss2'+' center;'+str(i))
        values_to_be_fitted.append(g.mu)
        values_to_be_fitted_what.append('gauss2'+' width;'+str(i))
        values_to_be_fitted.append(g.fwhm)
        #values_to_be_fitted_what.append('gauss2'+' assym;'+str(i))
        #values_to_be_fitted.append(g.alpha)
        values_to_be_fitted_what.append('gauss2'+' intens;'+str(i))
        print('intensity here',g.intensity)
        values_to_be_fitted.append(g.intensity)
    #values_to_be_fitted_what.append('fwhm')
    #values_to_be_fitted.append(0.09)
    print('t before',t.shape)
    #print(test_fit.gauss_1)
    test_fit.init_gauss()
    test_fit.ptot_function(values_to_be_fitted,values_to_be_fitted_what)
    print(test_fit.gauss2[0].intensity,test_fit.gauss2[1].intensity)
    plt.figure()
    TRPES_inital=test_fit.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False).transpose()
    plt.pcolormesh(t,eV,test_fit.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False).transpose(),cmap='nipy_spectral')
    plt.show()
    print('comparing shapes',t.shape,eV.shape,TRPES.shape)
    plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_be_fitted,
                  args=(values_to_be_fitted_what,test_fit.bi_exp_decay_2D,[[t],[eV]],[TRPES]),full_output=True)
    print('recovered vs fitted values')
    
    print('tau1_orig',1.2,'tau_recovered',plsq[0])
    print('tau2_orig',2.4,'tau_recovered',plsq[1])
    
    plsqs,Deltas=test_fit.extract_Deltas_plsqs(plsq,cov,values_to_be_fitted_what,test_fit.bi_exp_decay_2D,[[t],[eV]],[TRPES],[],[])
    print(plsqs,Deltas)
    
    plt.figure()
    plt.subplot(3,1,1)
    #TRPES=np.reshape(t,(t.shape[0],1))*np.reshape(eV,(1,eV.shape[0]))
    plt.pcolormesh(t,eV,TRPES,cmap='nipy_spectral')
    plt.text(3,2,'original',color='w')
    plt.subplot(3,1,2)
    plt.pcolormesh(t,eV,TRPES_inital,cmap='nipy_spectral')
    plt.text(3,2,'inital')
    plt.subplot(3,1,3)
    plt.pcolormesh(t,eV,test_fit.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False).transpose(),cmap='nipy_spectral')
    plt.text(3,2,'reconstructed')

    plt.figure()
    plt.title('DAS')
    plt.plot(eV,test_fit.sigma_1(eV).transpose()[:,0],label='DAS1')
    plt.plot(eV,simulated.sigma_1(eV).transpose()[:,0],ls='--',label='DAS1_original')
    plt.plot(eV,test_fit.sigma_2(eV)[0,:],label='DAS2')    
    plt.plot(eV,simulated.sigma_2(eV).transpose()[:,0],ls='--',label='DAS2_original')
    plt.legend()

    plt.figure()
    plt.title('decays')
    plt.plot(t,test_fit.mono_exp_decay(t,0,eV),label='t1')
    plt.plot(t,test_fit.bi_exp_decay_population(t,0,eV),label='t2')    
    plt.plot(t,simulated.mono_exp_decay(t,0,eV),ls='--',label='t1_orig')
    plt.plot(t,simulated.bi_exp_decay_population(t,0,eV),ls='--',label='t2_orig')
    plt.plot(t,test_fit.bi_exp_decay(t,0,eV),label='global')
    plt.legend()

    plt.figure()
    plt.title('residual')
    resid=TRPES-test_fit.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False).transpose()
    plt.pcolormesh(t,eV,resid,cmap='nipy_spectral')
    plt.show()

    
    #fitting them

    #worked?



