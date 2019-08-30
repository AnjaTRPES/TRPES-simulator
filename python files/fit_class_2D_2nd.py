import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

from fit_class2nd import fits_2nd
#testing an offset function I derived
#all times in fs

class fits_2D_2nd(object):
    fwhm=[0.08]
    time_offset=[0.]
    sigma_offset=[]
    tau1=120
    tau2=2400
    tau3=3600
    sigma_1=[]
    sigma_2=[]
    sigma_3=[]
    moy=[]
    counter=0
    interm_fit=fits_2nd()
    reporter=[]
    cov_DASs=[]
    

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
    
    def offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))

    def special_erf(self,t,t1,FWHM):
        sigma=self.fwhm_to_sigma(FWHM)
        lam=self.half_time_to_lambda(t1)
        try:
            g=np.nan_to_num(np.exp(0.5*(lam*sigma)**2-lam*t)*sp.special.erfc((-t+lam*sigma**2)/(np.sqrt(2)*sigma)))
        except FloatingPointError:
            #print('encountered floating point error,fit_class_2D_2nd')
            g=np.zeros(t.shape)
        return g
    
    def mono_exp_decay(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.special_erf(t,self.tau1,fwhm)

    def mono_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return (sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm))))-self.mono_exp_decay(t,n)
    
    def mono_exp_decay_with_offset(self,t,n,several_fwhms=False):
        return self.mono_exp_decay(t,n,several_fwhms)+self.offset(t,n)
    
    def mono_exp_decay_with_offset_final(self,t,n,several_fwhms=False):
        return self.mono_exp_decay(t,n,several_fwhms)+self.mono_exp_decay_final_state_pop(t,n,several_fwhms)
    
    def bi_exp_decay_population(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return (lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))
   
    def bi_exp_decay(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return self.mono_exp_decay(t,n,several_fwhms)+(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,several_fwhms)
        c=self.bi_exp_decay_population(t,n,several_fwhms)
        return (a-b-c)
    
    def bi_exp_decay_with_offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return self.mono_exp_decay_with_offset(t,n,several_fwhms)+self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_with_offset_final(self,t,n,several_fwhms=False):
        return self.bi_exp_decay(t,n,several_fwhms)+self.bi_exp_decay_final_state_pop(t,n,several_fwhms)
    
    def tri_exp_decay_population(self,t,n,several_fwhms=False):
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
    
    def tri_exp_decay(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]        
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        lam3=self.half_time_to_lambda(self.tau3)
        a=self.bi_exp_decay(t,n,several_fwhms)
        b1=(self.special_erf(t,self.tau1,fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2,fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3,fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,several_fwhms)
        c=self.bi_exp_decay_population(t,n,several_fwhms)
        d=self.tri_exp_decay_population(t,n,several_fwhms)
        return (a-b-c-d)

    def tri_exp_decay_with_offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        lam3=self.half_time_to_lambda(self.tau3)
        a=self.bi_exp_decay_with_offset(t,n,several_fwhms)
        b1=(self.special_erf(t,self.tau1,fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2,fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3,fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_with_offset_final(self,t,n,several_fwhms=False):
        return self.tri_exp_decay(t,n,several_fwhms)+self.tri_exp_decay_final_state_pop(t,n,several_fwhms)
         
    def mono_exp_parallel(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)
    def mono_exp_parallel_offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)+self.offset(t,n)
    def bi_exp_parallel(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2,fwhm)
    def bi_exp_parallel_offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2,fwhm)+self.offset(t,n)
    def tri_exp_parallel(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2,fwhm)+self.sigma_3[n]*self.special_erf(t,self.tau3,fwhm)
    def tri_exp_parallel_offset(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2,fwhm)+self.sigma_3[n]*self.special_erf(t,self.tau3,fwhm)+self.offset(t,n)
    def bi_exp_parallel_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_2[n]*self.special_erf(t,self.tau2,fwhm)
    def tri_exp_parallel_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_3[n]*self.special_erf(t,self.tau3,fwhm)
    
    def fit_function(self,Ptot,what_Ptots,function,tfit,Yfit,interm_fct,interm_fct_what,floating_t0=False,parallel=False,sigmas=None,several_fwhms=False):
        '''
        Fit function (leastsq fit) with which the exp. values are fitted.
        Ptot: values to fit
        what_Ptots: In string form, to attribute it to the class variables
        function: Fitfunction to use
        tfit: list of [exp times]
        Yfit: list of [exp. Signals]
        '''
        #print 'fitting %i functions',len(tfit)
        self.ptot_function(Ptot,what_Ptots)
        return_roots=[]
        for variable in [self.time_offset,self.sigma_offset,self.sigma_1,self.sigma_2,self.sigma_3,tfit,Yfit,self.fwhm]:
            self.check_whether_list(variable)
        #initializing test_fit:
        self.interm_fit.tau1=self.tau1
        self.interm_fit.tau2=self.tau2
        self.interm_fit.tau3=self.tau3
        self.interm_fit.fwhm=self.fwhm

        #for the uncertainties of the DAS
        self.cov_DASs=[]
        self.plsq_DASs=[]
        print(len(self.reporter))
        print('Ptot',Ptot,'what_Ptots',what_Ptots)
        for n in range(len(tfit)):
            if floating_t0==False:
                t=tfit[n]+self.time_offset[0]
            else:
                t=tfit[n]
            y=Yfit[n]
            #fitting the sigma parameters to the current selected tau parameters
            Ptot_interm,Ptot_interm_what=self.get_Ptot_interm(interm_fct_what,n,floating_t0,parallel)
            plsq,cov,info,msg,ier=leastsq(self.interm_fit.fit_function,Ptot_interm,
                              args=(Ptot_interm_what,interm_fct,[t],[y],floating_t0,sigmas[n]),full_output=True)
            self.reset_sigmas(plsq,interm_fct_what,n,floating_t0,parallel)
            self.cov_DASs.append(cov)
            self.plsq_DASs.append(plsq)
            #attributing the parameters
            if type(sigmas[n])==type(None):
                return_roots.append(np.sqrt(np.power(interm_fct(t,0)+self.moy[n]-Yfit[n],2)))
            else:
                return_roots.append(np.sqrt(np.power(np.divide(interm_fct(t,0)+self.moy[n]-Yfit[n],sigmas[n]),2)))
        if len(return_roots)==1:
            return return_roots[0]
        else:
            start=return_roots[0]
            for n in range(1,len(return_roots)):
                start=np.concatenate((start,np.nan_to_num(return_roots[n],nan=0.0,posinf=0.0,neginf=0.0)))
            self.reporter.append(np.sum(start))
            print('error',self.reporter[-1])
            return start
    
    
    def get_Ptot_interm(self,function,n,floating_t0,parallel):
        self.interm_fit.moy=[self.moy[n]]
        Ptot_interm=[]
        Ptot_interm_what=[]
        if floating_t0==True:
            self.interm_fit.time_offset=[self.time_offset[n]]
            Ptot_interm.append(self.time_offset[n])
            Ptot_interm_what.append('time_offset')
        if parallel==False:
            if 'mono_exp_decay' in function:
                self.interm_fit.sigma_1=[np.abs(self.sigma_1[n])]
                Ptot_interm.append(np.abs(self.sigma_1[n]))
                Ptot_interm_what.append('sigma_1')            
            elif 'bi_exp_decay' in function:
                self.interm_fit.sigma_1=[np.abs(self.sigma_1[n])]
                self.interm_fit.sigma_2=[np.abs(self.sigma_2[n])]
                Ptot_interm.append(np.abs(self.sigma_1[n]))
                Ptot_interm.append(np.abs(self.sigma_2[n]))
                Ptot_interm_what.append('sigma_1')
                Ptot_interm_what.append('sigma_2')
            elif 'tri_exp_decay' in function:
                self.interm_fit.sigma_1=[np.abs(self.sigma_1[n])]
                self.interm_fit.sigma_2=[np.abs(self.sigma_2[n])]
                self.interm_fit.sigma_2=[np.abs(self.sigma_3[n])]
                Ptot_interm.append(np.abs(self.sigma_1[n]))
                Ptot_interm.append(np.abs(self.sigma_2[n]))
                Ptot_interm.append(np.abs(self.sigma_3[n]))
                Ptot_interm_what.append('sigma_1')
                Ptot_interm_what.append('sigma_2')
                Ptot_interm_what.append('sigma_3')               
            if 'offset' in function:
                self.interm_fit.sigma_offset=[np.abs(self.sigma_offset[n])]
                Ptot_interm.append(np.abs(self.sigma_offset[n]))
                Ptot_interm_what.append('sigma_offset')    
            if 'fct_auto_corr' in function:
                self.interm_fit.sigma_1=[np.abs(self.sigma_1[n])]
                Ptot_interm.append(np.abs(self.sigma_1[n]))
                Ptot_interm_what.append('sigma_1') 
        else:
            if 'mono_exp_decay' in function:
                self.interm_fit.sigma_1=[self.sigma_1[n]]
                Ptot_interm.append(self.sigma_1[n])
                Ptot_interm_what.append('sigma_1')            
            elif 'bi_exp_decay' in function:
                self.interm_fit.sigma_1=[self.sigma_1[n]]
                self.interm_fit.sigma_2=[self.sigma_2[n]]
                Ptot_interm.append(self.sigma_1[n])
                Ptot_interm.append(self.sigma_2[n])
                Ptot_interm_what.append('sigma_1')
                Ptot_interm_what.append('sigma_2')
            elif 'tri_exp_decay' in function:
                self.interm_fit.sigma_1=[self.sigma_1[n]]
                self.interm_fit.sigma_2=[self.sigma_2[n]]
                self.interm_fit.sigma_2=[self.sigma_3[n]]
                Ptot_interm.append(self.sigma_1[n])
                Ptot_interm.append(self.sigma_2[n])
                Ptot_interm.append(self.sigma_3[n])
                Ptot_interm_what.append('sigma_1')
                Ptot_interm_what.append('sigma_2')
                Ptot_interm_what.append('sigma_3')               
            if 'offset' in function:
                self.interm_fit.sigma_offset=[self.sigma_offset[n]]
                Ptot_interm.append(self.sigma_offset[n])
                Ptot_interm_what.append('sigma_offset')    
            if 'fct_auto_corr' in function:
                self.interm_fit.sigma_1=[self.sigma_1[n]]
                Ptot_interm.append(self.sigma_1[n])
                Ptot_interm_what.append('sigma_1') 
        return Ptot_interm,Ptot_interm_what
        
    def reset_sigmas(self,ptots_interm,function,n,floating_t0,parallel):
        g=0
        if floating_t0==True:
            self.time_offset[n]=ptots_interm[0]
            g+=1
        if parallel==False:
            if 'mono_exp_decay' in function:
                self.sigma_1[n]=np.abs(ptots_interm[g])
            elif 'bi_exp_decay' in function:
                self.sigma_1[n]=np.abs(ptots_interm[g])
                self.sigma_2[n]=np.abs(ptots_interm[g+1])
            elif 'tri_exp_decay' in function:
                self.sigma_1[n]=np.abs(ptots_interm[g])
                self.sigma_2[n]=np.abs(ptots_interm[g+1])
                self.sigma_3[n]=np.abs(ptots_interm[g+2])
            if 'offset' in function:
                self.sigma_offset[n]=np.abs(ptots_interm[-1])
            if 'fct_auto_corr' in function:
                self.sigma_1[n]=np.abs(ptots_interm[g])
        else:
            if 'mono_exp_decay' in function:
                self.sigma_1[n]=ptots_interm[g]
            elif 'bi_exp_decay' in function:
                self.sigma_1[n]=ptots_interm[g]
                self.sigma_2[n]=ptots_interm[g+1]
            elif 'tri_exp_decay' in function:
                self.sigma_1[n]=ptots_interm[g]
                self.sigma_2[n]=ptots_interm[g+1]
                self.sigma_3[n]=ptots_interm[g+2]
            if 'offset' in function:
                self.sigma_offset[n]=ptots_interm[-1]
            if 'fct_auto_corr' in function:
                self.sigma_1[n]=ptots_interm[g]
    
    def ptot_function(self,Ptot,whatPtots):
        '''
        with whatPtots the values in Ptot will be attributed to the values in the class
        whatPtots: list of stringex: [time_offset,fwhm,...]
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
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t)**2)/(2*(sigma)**2))
    def fct_auto_corr_with_offset(self,t,n,several_fwhms=False):
        sigma=self.fwhm_to_sigma(self.fwhm[n])
        return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t)**2)/(2*(sigma)**2))+self.offset(t,n)

    def get_deltas_DAS(self,plsq,tfit,Yfit,interm_fct,interm_fct_what, parallel=False,floating_t0=False,sigmas=None,several_fwhms=False):
        """
        This function gets the uncertainties for each DAS
        """
        DAS_plsqs=[]
        DAS_deltas=[]
        for n in range(len(tfit)):
            if floating_t0==False:
                t=tfit[n]+self.time_offset[0]
            else:
                t=tfit[n]
            y=Yfit[n]
            Ptot_interm,Ptot_interm_what=self.get_Ptot_interm(interm_fct_what,n,floating_t0,parallel)
            #get the residual
            Yres=self.interm_fit.fit_function(Ptot_interm,Ptot_interm_what,interm_fct,[t],[y],floating_t0,sigmas[n])
            res=np.sqrt(np.power(Yres,2).sum()/Yres.size)
            plsqs=[]
            Deltas=[]
            for i in range(len(self.plsq_DASs[n])):
                plsqs.append(self.plsq_DASs[i])
                if type(self.cov_DASs[n])==type(None):
                    Delta=float('nan')
                elif self.cov_DASs[n].any()==None:
                    Delta=float('nan')
                else:
                    Delta=(np.sqrt(self.cov_DASs[n].diagonal())*res)[i]
                Deltas.append(Delta)
            DAS_plsqs.append(plsqs)
            DAS_deltas.append(Deltas)
        return DAS_plsqs,DAS_deltas

        
     
    def extract_Deltas_plsqs(self,plsq,cov,whatPtots,function,tfit,Yfit,interm_fct,interm_fct_what,parallel=False,sigmas=None,several_fwhms=False):
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
        Yres=self.fit_function(plsq,whatPtots,function,tfit,Yfit,interm_fct,interm_fct_what,parallel=parallel,sigmas=sigmas)
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
                Delta=(np.sqrt(cov.diagonal()*res))[n]
            Deltas.append(Delta)
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
    #makint test case
    from fit_class_2D import fits_2D
    from skewed_gauss import skewed_gauss
    eV=np.arange(0,5,0.01)
    t=np.arange(-2,15,0.05)
    #simulating one 2D-TRPES
    simulated=fits_2D()
    gauss1=[skewed_gauss(0,0.2,0.,1.)]
    #        skewed_gauss(1,0.5,0.,1.)]
    gauss2=[skewed_gauss(2,0.2,0.,0.2),
            skewed_gauss(0.5,0.5,0.,0.2)]
    simulated.gauss1=gauss1
    simulated.gauss2=gauss2
    simulated.tau1=0.8
    simulated.tau2=5.
    TRPES=simulated.bi_exp_decay_2D(np.reshape(t,(t.shape[0],1)),0,eV,several_fwhms=False)
    TRPES=TRPES
    #TRPES=TRPES.transpose()

    #alright, making 'my' testobject
    test_fit=fits_2D_2nd()
    #alright, starting values
    values_to_fit_what=['tau1','tau2']
    values_to_fit=[0.6,4.]

    values_to_fit_2=[]
    values_to_fit_what_2=[]
    for n in range(TRPES.shape[1]):
        values_to_fit_2.append(TRPES[:,n].max())
        values_to_fit_what_2.append('sigma_1')
        values_to_fit_2.append(TRPES[:,n].max())
        values_to_fit_what_2.append('sigma_2')
        values_to_fit_2.append(0.)
        values_to_fit_what_2.append('moy')
    #prepping it
    test_fit.ptot_function(values_to_fit_2,values_to_fit_what_2)
    #getting tfits and yfits
    tfits=[]
    yfits=[]
    for n in range(TRPES.shape[1]):
        tfits.append(t)
        yfits.append(TRPES[:,n])
    
    import matplotlib.pyplot as plt
    '''
    plt.figure()
    plt.plot(tfits[0],yfits[0])
    plt.show()
    '''
    plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_fit,
                              args=(values_to_fit_what,test_fit.bi_exp_decay,tfits,yfits,
                                    test_fit.interm_fit.bi_exp_decay,'bi_exp_decay'),full_output=True)
    plsqs,Deltas=test_fit.extract_Deltas_plsqs(plsq,cov,values_to_fit_what,test_fit.bi_exp_decay,tfits,yfits,
                                               test_fit.interm_fit.bi_exp_decay,'bi_exp_decay') 

    #getting the decay associated spectra out
    DAS_1=np.array(test_fit.sigma_1)
    DAS_2=np.array(test_fit.sigma_2)

    pop_1=test_fit.mono_exp_decay(t,0)
    #pop_1=pop_1/max(pop_1)
    pop_2=test_fit.bi_exp_decay_population(t,0)
    #pop_2=pop_2/max(pop_2)

    TRPES_1=np.reshape(DAS_1,(1,DAS_1.shape[0]))*np.ones((pop_1.shape[0],DAS_1.shape[0]))*np.reshape(pop_1,(pop_1.shape[0],1))
    TRPES_2=np.reshape(DAS_2,(1,DAS_2.shape[0]))*np.ones((pop_2.shape[0],DAS_2.shape[0]))*np.reshape(pop_2,(pop_2.shape[0],1))
    TRPES_rec=TRPES_1+TRPES_2    
    #showing the results
    plt.figure()
    plt.subplot(3,1,1)
    plt.pcolormesh(t,eV,TRPES.transpose())
    plt.xlabel('original')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.pcolormesh(t,eV,TRPES_rec.transpose())
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.pcolormesh(t,eV,TRPES.transpose()-TRPES_rec.transpose())
    plt.colorbar()

    plt.figure()
    plt.subplot(4,1,1)
    plt.pcolormesh(t,eV,TRPES_1.transpose())
    plt.subplot(4,1,2)
    plt.pcolormesh(t,eV,TRPES_2.transpose())
    plt.subplot(4,1,3)
    plt.pcolormesh(t,eV,(TRPES_1+TRPES_2).transpose())
    plt.subplot(4,1,4)
    plt.pcolormesh(t,eV,TRPES.transpose())
    
    plt.figure()
    plt.plot(eV,DAS_1,label='DAS1_rec')
    plt.plot(eV,DAS_2,label='DAS2_rec')
    plt.plot(eV,gauss1[0].calculate_gauss(eV),'--',label='DAS1_orig')
    plt.plot(eV,gauss2[0].calculate_gauss(eV)+gauss2[1].calculate_gauss(eV),'--',label='DAS1_orig')
    plt.legend()
    plt.show()

    

        






