import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#testing an offset function I derived
#all times in fs

class fits_bidir(object):

    def __init__(self,test_param=20):
        self.fwhm=0.08
        self.time_offset=0.
        self.sigma_offset=[1.,1.]
        self.tau1=[2.12,2.12]
        self.tau2=[2.4,2.9]
        self.tau3=[3.6,4.]
        self.sigma_1=[1.,1.]
        self.sigma_2=[0.5,0.5]
        self.sigma_3=[0.3,0.3]
        self.moy=0.
        self.counter=0
        self.reporter=0
    
    def fwhm_to_sigma(self,fwhm):
        return fwhm/(2*np.sqrt(2*np.log(2)))

    def half_time_to_lambda(self,half_time):
        return 1./half_time
    
    def offset(self,t,n):
        return self.sigma_offset[n]*sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))

    def special_erf(self,t,t1,FWHM):
        sigma=self.fwhm_to_sigma(FWHM)
        lam=self.half_time_to_lambda(t1)
        try:
            g=np.nan_to_num(np.exp(0.5*(lam*sigma)**2-lam*t)*sp.special.erfc((-t+lam*sigma**2)/(np.sqrt(2)*sigma)))
        except FloatingPointError:
            #print('encountered floating point error,bidir')
            g=np.zeros(t.shape)
        return g
    
    def mono_exp_decay(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)

    def mono_exp_decay_final_state_pop(self,t,n):
        return self.sigma_offset[n]*((sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm))))-self.mono_exp_decay(t,n)/self.sigma_1[n])
    
    def mono_exp_decay_with_offset(self,t,n):
        return self.mono_exp_decay(t,n)+self.offset(t,n)
    
    def mono_exp_decay_with_offset_final(self,t,n):
        return self.mono_exp_decay(t,n)+self.mono_exp_decay_final_state_pop(t,n)
    
    def bi_exp_decay_population(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))
   
    def bi_exp_decay(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return self.mono_exp_decay(t,n)+self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))

    def bi_exp_decay_final_state_pop(self,t,n):
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))
        b=self.mono_exp_decay(t,n)/self.sigma_1[n]
        c=self.bi_exp_decay_population(t,n)/self.sigma_2[n]
        return self.sigma_offset[n]*(a-b-c)
    
    def bi_exp_decay_with_offset(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return self.mono_exp_decay_with_offset(t,n)+self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))

    def bi_exp_decay_with_offset_final(self,t,n):
        return self.bi_exp_decay(t,n)+self.bi_exp_decay_final_state_pop(t,n)
    
    def tri_exp_decay_population(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        return self.sigma_3[n]*lam1*lam2*(b1+b3)  
    
    def tri_exp_decay(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        a=self.bi_exp_decay(t,n)
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=self.sigma_3[n]*lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_final_state_pop(self,t,n):
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))
        b=self.mono_exp_decay(t,n)/self.sigma_1[n]
        c=self.bi_exp_decay_population(t,n)/self.sigma_2[n]
        d=self.tri_exp_decay_population(t,n)/self.sigma_3[n]
        return self.sigma_offset[n]*(a-b-c-d)

    def tri_exp_decay_with_offset(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        a=self.bi_exp_decay_with_offset(t,n)
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=self.sigma_3[n]*lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_with_offset_final(self,t,n):
        return self.tri_exp_decay(t,n)+self.tri_exp_decay_final_state_pop(t,n)
     
    def mono_exp_parallel(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)
    def mono_exp_parallel_offset(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)+self.offset(t,n)
    def bi_exp_parallel(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2[n],self.fwhm)
    def bi_exp_parallel_offset(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2[n],self.fwhm)+self.offset(t,n)
    def tri_exp_parallel(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2[n],self.fwhm)+self.sigma_3[n]*self.special_erf(t,self.tau3[n],self.fwhm)
    def tri_exp_parallel_offset(self,t,n):
        return self.sigma_1[n]*self.special_erf(t,self.tau1[n],self.fwhm)+self.sigma_2[n]*self.special_erf(t,self.tau2[n],self.fwhm)+self.sigma_3[n]*self.special_erf(t,self.tau3[n],self.fwhm)+self.offset(t,n)
    def bi_exp_parallel_pop(self,t,n):
        return self.sigma_2[n]*self.special_erf(t,self.tau2[n],self.fwhm)
    def tri_exp_parallel_pop(self,t,n):
        return self.sigma_3[n]*self.special_erf(t,self.tau3[n],self.fwhm)
    
    def fit_function(self,Ptot,what_Ptots,function_pos,function_min,tfit,Yfit,floating_t0=False):
        '''
        Fit function (leastsq fit) with which the exp. values are fitted.
        Ptot: values to fit
        what_Ptots: In string form, to attribute it to the class variables
        function_pos: Fitfunction to use for the positive direction
        function_min: Fitfunction to use for the negative direction
        tfit=exp times
        Yfit: exp. Signals
        '''
        #print 'fitting %i functions',len(tfit)
        self.ptot_function(Ptot,what_Ptots)
        return_roots=[]
        t=tfit#+self.time_offset
        if floating_t0==True:
            t=t+self.time_offset
        #getting the positive side
        Ypos=function_pos(t,0)
        Ymin=function_min(-t,1)
        Y=Ypos+Ymin
        self.reporter+=1
        return_roots=np.sqrt(np.power(Y+self.moy-Yfit,2))
        return return_roots


    
    def ptot_function(self,Ptot,whatPtots):
        '''
        with whatPtots the values in Ptot will be attributed to the values in the class
        whatPtots: list of stringex: [time_offset,fwhm,...]
        '''
        #print('Ptot',Ptot)
        #print('whatPtots',whatPtots)
        if len(Ptot)!=len(whatPtots):
            print ('len(Ptot)!=len(whatPtots)')
            #raise WhatPtotsError Need to look up how to define my own errors
        for n in range(len(whatPtots)):
            if whatPtots[n]=='fwhm':
                self.fwhm=Ptot[n]
            if whatPtots[n]=='time_offset':
                self.time_offset=Ptot[n]
            if 'sigma_offset' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_offset[0]=Ptot[n]
                elif '_min' in whatPtots[n]:
                    self.sigma_offset[1]=Ptot[n]
            if 'tau1'in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.tau1[0]=abs(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.tau1[1]=abs(Ptot[n])
            if 'tau2' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.tau2[0]=abs(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.tau2[1]=abs(Ptot[n])
            if 'tau3'in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.tau3[0]=abs(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.tau3[1]=abs(Ptot[n])
            if'sigma_1' in  whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_1[0]=Ptot[n]
                elif '_min' in whatPtots[n]:
                    self.sigma_1[1]=Ptot[n]
            if 'sigma_2' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_2[0]=Ptot[n]
                elif '_min' in whatPtots[n]:
                    self.sigma_2[1]=Ptot[n]
            if 'sigma_3' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_3[0]=Ptot[n]
                elif '_min' in whatPtots[n]:
                    self.sigma_3[1]=Ptot[n]
            if whatPtots[n]=='moy':
                self.moy=Ptot[n]


                
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
     
    def extract_Deltas_plsqs(self,plsq,cov,whatPtots,function_pos,function_min,tfit,Yfit):
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
        Yres=self.fit_function(plsq,whatPtots,function_pos,function_min,tfit,Yfit)
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
    print('testing this class for bidriectional fits')
    import matplotlib.pyplot as plt
    t=np.arange(-5,15,0.05)
    #get a sample didirectional decay
    sample_fits=fits_bidir()
    y1=sample_fits.bi_exp_decay(t,0)
    y2=sample_fits.mono_exp_decay(-t,1)
    noise=np.random.normal(0, 0.2, t.shape)
    y=y1+y2+noise
    plt.figure()
    plt.plot(t,y1)
    plt.plot(t,y2)
    plt.plot(t,y)
    plt.show()
    
    #make initial guesses
    test_fit=fits_bidir()
    values_to_fit=[0.12,0.2,4.4,
                   2.,2.,0.2]
    values_to_fit_what=['tau1_min','tau1_pos','tau2_pos',
                        'sigma_1_pos','sigma_1_min','sigma_2_pos']
    function_pos=test_fit.bi_exp_decay
    function_min=test_fit.mono_exp_decay
    test_fit.ptot_function(values_to_fit,values_to_fit_what)
    print('values before fitting',test_fit.fwhm,test_fit.tau1,test_fit.tau2,test_fit.sigma_1,test_fit.sigma_2,test_fit.time_offset)
    plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_fit,
                  args=(values_to_fit_what,function_pos,function_min,t,y),full_output=True)
    #result as a plot
    y_result=function_pos(t,0)+function_min(-t,1)
    plt.figure()
    plt.plot(t,y,'k')
    plt.plot(t,y1+y2,'k')
    plt.plot(t,y_result,'r--')
    plt.show()
    plsqs,Deltas=test_fit.extract_Deltas_plsqs(plsq,cov,values_to_fit_what,function_pos,function_min,t,y)
    for n in range(len(plsqs)):
        print(values_to_fit_what[n]+'='+'{:.3f}'.format(plsqs[n])+'+-'+'{:.3f}'.format(Deltas[n]))
    
    #print(test_fit.extract_Deltas_plsqs(plsq,cov,values_to_fit_what,function_pos,function_min,t,y))
    print('original values',sample_fits.fwhm,sample_fits.tau1,sample_fits.tau2,sample_fits.sigma_1,sample_fits.sigma_2)
    #print('retrieved values',test_fit.fwhm,test_fit.tau1,test_fit.tau2,test_fit.sigma_1,test_fit.sigma_2,test_fit.time_offset)
    
