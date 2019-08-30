import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#testing an offset function I derived
#all times in fs
from fit_class_bidir import fits_bidir

class fits_bidir_2d(object):

    def __init__(self,test_param=20):
        self.fwhm=0.08
        self.time_offset=[]
        self.sigma_offset=[[],[]]
        self.tau1=[2.12,2.12]
        self.tau2=[2.4,2.9]
        self.tau3=[3.6,4.]
        self.sigma_1=[[],[]]
        self.sigma_2=[[],[]]
        self.sigma_3=[[],[]]
        self.moy=[]
        self.counter=0
        self.reporter=[]
        self.interm_fit=fits_bidir()
        self.cov_DASs=[]
        self.plsq_DASs=[]
    
    def fwhm_to_sigma(self,fwhm):
        return fwhm/(2*np.sqrt(2*np.log(2)))

    def half_time_to_lambda(self,half_time):
        return 1./half_time
    
    def offset(self,t,n):
        return sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))

    def special_erf(self,t,t1,FWHM):
        sigma=self.fwhm_to_sigma(FWHM)
        lam=self.half_time_to_lambda(t1)
        try:
            g=np.nan_to_num(np.exp(0.5*(lam*sigma)**2-lam*t)*sp.special.erfc((-t+lam*sigma**2)/(np.sqrt(2)*sigma)))
        except FloatingPointError:
            g=np.zeros(t.shape)
        return g
    
    def mono_exp_decay(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)

    def mono_exp_decay_final_state_pop(self,t,n):
        return (sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm))))-self.mono_exp_decay(t,n)
    
    def mono_exp_decay_with_offset(self,t,n):
        return self.mono_exp_decay(t,n)+self.offset(t,n)
    
    def mono_exp_decay_with_offset_final(self,t,n):
        return self.mono_exp_decay(t,n)+self.mono_exp_decay_final_state_pop(t,n)
    
    def bi_exp_decay_population(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return (lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))
   
    def bi_exp_decay(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return self.mono_exp_decay(t,n)+(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))

    def bi_exp_decay_final_state_pop(self,t,n):
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))
        b=self.mono_exp_decay(t,n)
        c=self.bi_exp_decay_population(t,n)
        return (a-b-c)
    
    def bi_exp_decay_with_offset(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        return self.mono_exp_decay_with_offset(t,n)+(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1[n],self.fwhm)-self.special_erf(t,self.tau2[n],self.fwhm))

    def bi_exp_decay_with_offset_final(self,t,n):
        return self.bi_exp_decay(t,n)+self.bi_exp_decay_final_state_pop(t,n)
    
    def tri_exp_decay_population(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        return lam1*lam2*(b1+b3)  
    
    def tri_exp_decay(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        a=self.bi_exp_decay(t,n)
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_final_state_pop(self,t,n):
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(self.fwhm)))
        b=self.mono_exp_decay(t,n)
        c=self.bi_exp_decay_population(t,n)
        d=self.tri_exp_decay_population(t,n)
        return (a-b-c-d)

    def tri_exp_decay_with_offset(self,t,n):
        lam1=self.half_time_to_lambda(self.tau1[n])
        lam2=self.half_time_to_lambda(self.tau2[n])
        lam3=self.half_time_to_lambda(self.tau3[n])
        a=self.bi_exp_decay_with_offset(t,n)
        b1=(self.special_erf(t,self.tau1[n],self.fwhm)/(lam3-lam1)-self.special_erf(t,self.tau2[n],self.fwhm)/(lam3-lam2))/(lam2-lam1)
        b2=self.special_erf(t,self.tau3[n],self.fwhm)
        b3=b2/((lam3-lam1)*(lam3-lam2))
        b=lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_with_offset_final(self,t,n):
        return self.tri_exp_decay(t,n)+self.tri_exp_decay_final_state_pop(t,n)
    
    def mono_exp_parallel(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)
    def mono_exp_parallel_offset(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)+self.offset(t,n)
    def bi_exp_parallel(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)+self.special_erf(t,self.tau2[n],self.fwhm)
    def bi_exp_parallel_offset(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)+self.special_erf(t,self.tau2[n],self.fwhm)+self.offset(t,n)
    def tri_exp_parallel(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)+self.special_erf(t,self.tau2[n],self.fwhm)+self.special_erf(t,self.tau3[n],self.fwhm)
    def tri_exp_parallel_offset(self,t,n):
        return self.special_erf(t,self.tau1[n],self.fwhm)+self.special_erf(t,self.tau2[n],self.fwhm)+self.special_erf(t,self.tau3[n],self.fwhm)+self.offset(t,n)
    def bi_exp_parallel_pop(self,t,n):
        return self.special_erf(t,self.tau2[n],self.fwhm)
    def tri_exp_parallel_pop(self,t,n):
        return self.special_erf(t,self.tau3[n],self.fwhm)
    
    def fit_function(self,Ptot,what_Ptots,function_pos,function_min,tfit,Yfit,floating_t0=False,parallel_pos=False,parallel_min=False,sigmas=None):
        '''
        Fit function (leastsq fit) with which the exp. values are fitted.
        Ptot: values to fit
        what_Ptots: In string form, to attribute it to the class variables
        function_pos: Fitfunction to use for the positive direction
        function_min: Fitfunction to use for the negative direction
        tfit=[exp times]
        Yfit: [exp. Signals]
        '''
        self.ptot_function(Ptot,what_Ptots)
        print(Ptot,what_Ptots)
        return_roots=[]
        self.interm_fit.tau1=self.tau1
        self.interm_fit.tau2=self.tau2
        self.interm_fit.tau3=self.tau3
        self.interm_fit.fwhm=self.fwhm
        self.cov_DASs=[]
        self.plsq_DASs=[]
        for k,t in enumerate(tfit):
            self.interm_fit.moy=self.moy[k]
            if floating_t0==True:
                self.interm_fit.time_offset=self.time_offset[k]
            t=tfit[k]+self.time_offset[0]
            Maxcalls=250
            if floating_t0==True:
                t=tfit[k]       
                Maxcalls=10000
            y=Yfit[k]
            Ptot_interm,Ptot_interm_what=self.get_Ptot_interm(function_pos,function_min,k,floating_t0,parallel_pos,parallel_min)
            #plsq,cov=leastsq(self.interm_fit.fit_function,Ptot_interm,
            #     args=(Ptot_interm_what,function_pos,function_min,t,y,floating_t0),full_output=False,maxfev=Maxcalls)
            #print('sigmas[k]',sigmas[k])
            #print(type(sigmas[k]))
            plsq,cov,info,msg,ier=leastsq(self.interm_fit.fit_function,Ptot_interm,
                                          args=(Ptot_interm_what,function_pos,function_min,t,y,floating_t0,sigmas[k]),full_output=True)
            self.cov_DASs.append(cov)
            self.plsq_DASs.append(plsq)
            self.reset_sigmas(plsq,Ptot_interm_what,k,floating_t0,parallel_pos,parallel_min)
            Ypos=function_pos(t,0)
            Ymin=function_min(-t,1)            
            if type(sigmas[k])==type(None):
                return_roots.append(np.sqrt(np.power(Ypos+Ymin+self.moy[k]-Yfit[k],2)))
            else:                
                return_roots.append(np.sqrt(np.power(np.divide(Ypos+Ymin+self.moy[k]-Yfit[k],sigmas[k]),2)))
        if len(return_roots)==1:
            return return_roots[0]
        else:
            start=return_roots[0]
            for n in range(1,len(return_roots)):
                start=np.concatenate((start,return_roots[n]))
            self.reporter.append(np.sum(start))
            print(len(self.reporter))
            return start
    def get_Ptot_interm(self, function_pos,function_min,k,floating_t0,parallel_pos,parallel_min):
        which=['_pos','_min']
        Ptot_interm=[]
        Ptot_interm_what=[]
        for n,function in enumerate([str(function_pos),str(function_min)]):
            if [parallel_pos,parallel_min][n]==False:
                if 'mono_exp' in function:
                    Ptot_interm.append(np.abs(self.sigma_1[n][k]))
                    Ptot_interm_what.append('sigma_1'+which[n])
                elif 'bi_exp' in function:
                    Ptot_interm.append(np.abs(self.sigma_1[n][k]))
                    Ptot_interm_what.append('sigma_1'+which[n])
                    Ptot_interm.append(np.abs(self.sigma_2[n][k]))
                    Ptot_interm_what.append('sigma_2'+which[n])
                elif 'tri_exp' in function:
                    Ptot_interm.append(np.abs(self.sigma_1[n][k]))
                    Ptot_interm_what.append('sigma_1'+which[n])
                    Ptot_interm.append(np.abs(self.sigma_2[n][k]))
                    Ptot_interm_what.append('sigma_2'+which[n])
                    Ptot_interm.append(np.abs(self.sigma_3[n][k]))
                    Ptot_interm_what.append('sigma_3'+which[n])
                if 'offset' in function:
                    Ptot_interm.append(np.abs(self.sigma_offset[n][k]))
                    Ptot_interm_what.append('sigma_offset'+which[n])
            else:
                if 'mono_exp' in function:
                    Ptot_interm.append(self.sigma_1[n][k])
                    Ptot_interm_what.append('sigma_1'+which[n])
                elif 'bi_exp' in function:
                    Ptot_interm.append(self.sigma_1[n][k])
                    Ptot_interm_what.append('sigma_1'+which[n])
                    Ptot_interm.append(self.sigma_2[n][k])
                    Ptot_interm_what.append('sigma_2'+which[n])
                elif 'tri_exp' in function:
                    Ptot_interm.append(self.sigma_1[n][k])
                    Ptot_interm_what.append('sigma_1'+which[n])
                    Ptot_interm.append(self.sigma_2[n][k])
                    Ptot_interm_what.append('sigma_2'+which[n])
                    Ptot_interm.append(self.sigma_3[n][k])
                    Ptot_interm_what.append('sigma_3'+which[n])
                if 'offset' in function:
                    Ptot_interm.append(self.sigma_offset[n][k])
                    Ptot_interm_what.append('sigma_offset'+which[n])
        if floating_t0==True:
            Ptot_interm.append(self.time_offset[k])
            Ptot_interm_what.append('time_offset')
        else:
            self.interm_fit.time_offset=self.time_offset[0]
        return Ptot_interm, Ptot_interm_what

    def reset_sigmas(self,plsq,Ptot_interm_what,k,floating_t0,parallel_pos,parallel_min):

        for n in range(len(Ptot_interm_what)):
            if 'sigma_offset' in Ptot_interm_what[n]:
                if '_pos' in Ptot_interm_what[n]:
                    if parallel_pos==False:
                        self.sigma_offset[0][k]=np.abs(plsq[n])
                    else:
                        self.sigma_offset[0][k]=plsq[n]
                elif '_min' in Ptot_interm_what[n]:
                    if parallel_min==False:
                        self.sigma_offset[1][k]=np.abs(plsq[n])
                    else:
                        self.sigma_offset[1][k]=plsq[n]
            if'sigma_1' in  Ptot_interm_what[n]:
                if '_pos' in Ptot_interm_what[n]:
                    if parallel_pos==False:
                        self.sigma_1[0][k]=np.abs(plsq[n])
                    else:
                        self.sigma_1[0][k]=plsq[n]
                elif '_min' in Ptot_interm_what[n]:
                    if parallel_min==False:
                        self.sigma_1[1][k]=np.abs(plsq[n])
                    else:
                        self.sigma_1[1][k]=plsq[n]
            if 'sigma_2' in Ptot_interm_what[n]:
                if '_pos' in Ptot_interm_what[n]:
                    if parallel_pos==False:
                        self.sigma_2[0][k]=np.abs(plsq[n])
                    else:
                        self.sigma_2[0][k]=plsq[n]
                elif '_min' in Ptot_interm_what[n]:
                    if parallel_min==False:
                        self.sigma_2[1][k]=np.abs(plsq[n])
                    else:
                        self.sigma_2[1][k]=plsq[n]
            if 'sigma_3' in Ptot_interm_what[n]:
                if '_pos' in Ptot_interm_what[n]:
                    if parallel_pos==False:
                        self.sigma_3[0][k]=np.abs(plsq[n])
                    else:
                        self.sigma_3[0][k]=plsq[n]
                elif '_min' in Ptot_interm_what[n]:
                    if parallel_min==False:
                        self.sigma_3[1][k]=np.abs(plsq[n])
                    else:
                        self.sigma_3[1][k]=plsq[n]

        if floating_t0==True:
            self.time_offset[k]=plsq[-1]

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
        if 'sigma_offset' in whatPtots:
            self.sigma_offset=[[],[]]
        if 'sigma_1' in whatPtots:
            self.sigma_1=[[],[]]
        if 'sigma_2' in whatPtots:
            self.sigma_2=[[],[]]
        if 'sigma_3' in whatPtots:
            self.sigma_3=[[],[]]
        if 'moy' in whatPtots:
            self.moy=[]
        if 'time_offset' in whatPtots:
            self.time_offset=[]
        for n in range(len(whatPtots)):
            if whatPtots[n]=='fwhm':
                self.fwhm=Ptot[n]
            if whatPtots[n]=='time_offset':
                self.time_offset.append(Ptot[n])
            if 'sigma_offset' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_offset[0].append(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.sigma_offset[1].append(Ptot[n])
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
                    self.sigma_1[0].append(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.sigma_1[1].append(Ptot[n])
            if 'sigma_2' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_2[0].append(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.sigma_2[1].append(Ptot[n])
            if 'sigma_3' in whatPtots[n]:
                if '_pos' in whatPtots[n]:
                    self.sigma_3[0].append(Ptot[n])
                elif '_min' in whatPtots[n]:
                    self.sigma_3[1].append(Ptot[n])
            if whatPtots[n]=='moy':
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

    def get_deltas_DAS(self,plsq,cov,whatPtots,function_pos,function_min,tfit,Yfit,floating_t0,parallel_pos,parallel_min,sigmas):
        """
        This function gets the uncertainties for each DAS
        """

        DAS_plsqs=[]
        DAS_deltas=[]
        for k,t in enumerate(tfit):
            self.interm_fit.moy=self.moy[k]
            if floating_t0==True:
                self.interm_fit.time_offset=self.time_offset[k]
            t=tfit[k]+self.time_offset[0]
            Maxcalls=250
            if floating_t0==True:
                t=tfit[k]       
                Maxcalls=10000
            y=Yfit[k]
            Ptot_interm,Ptot_interm_what=self.get_Ptot_interm(function_pos,function_min,k,floating_t0,parallel_pos,parallel_min)
            #get the residual
            Yres=self.interm_fit.fit_function(Ptot_interm,Ptot_interm_what,function_pos,function_min,t,y,floating_t0,sigmas[k])
            res=np.sqrt(np.power(Yres,2).sum()/Yres.size)
            plsqs=[]
            Deltas=[]
            for i in range(len(self.plsq_DASs[k])):
                plsqs.append(self.plsq_DASs[i])
                if type(self.cov_DASs[k])==type(None):
                    Delta=float('nan')
                elif self.cov_DASs[k].any()==None:
                    Delta=float('nan')
                else:
                    Delta=(np.sqrt(self.cov_DASs[k].diagonal())*res)[i]
                Deltas.append(Delta)
            DAS_plsqs.append(plsqs)
            DAS_deltas.append(Deltas)
        return DAS_plsqs,DAS_deltas
        

        
     
         
    def extract_Deltas_plsqs(self,plsq,cov,whatPtots,function_pos,function_min,tfit,Yfit,floating_t0,parallel_pos,parallel_min,sigmas):
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
        Yres=self.fit_function(plsq,whatPtots,function_pos,function_min,tfit,Yfit,floating_t0,parallel_pos,parallel_min,sigmas=sigmas)
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
    print('testing this class for bidriectional fits')
    import matplotlib.pyplot as plt
    from skewed_gauss import skewed_gauss
    from fit_class_bidir import fits_bidir
    eV=np.arange(0,5,0.01)
    t=np.arange(-5,15,0.05)
    #simulating one 2D-TRPES
    gauss1=[skewed_gauss(0,0.2,0.,1.),
            skewed_gauss(1,0.5,0.,1.)]
    gauss2=[skewed_gauss(2,0.2,0.,0.2),
            skewed_gauss(0.5,0.5,0.,0.2)]
    gauss_min=[skewed_gauss(0.4,0.2,0.,0.4),
            skewed_gauss(1,0.5,0.,0.4)]
    gauss_pos_tau1=gauss1[0].calculate_gauss(eV)+gauss1[1].calculate_gauss(eV)
    gauss_pos_tau2=gauss2[0].calculate_gauss(eV)+gauss2[1].calculate_gauss(eV)
    gauss_min_tau1=gauss_min[0].calculate_gauss(eV)+gauss_min[1].calculate_gauss(eV)

    sample_fit=fits_bidir()
    sample_fit.tau1=[0.3,0.9]
    sample_fit.tau2=[4.3,2.8]
    pop1=sample_fit.mono_exp_decay(t,0)
    pop2=sample_fit.bi_exp_decay_population(t,0)
    pop3=sample_fit.mono_exp_decay(-t,1)
    ones=np.ones((eV.shape[0],t.shape[0]))
    pops=[pop1,pop2,pop3]
    DAS=[gauss_pos_tau1,gauss_pos_tau2,gauss_min_tau1]
    TRPES=np.zeros(ones.shape)
    for n,das in enumerate(DAS):
        TRPES+=np.reshape(das,(eV.shape[0],1))*ones*pops[n]
    #plotting it
    plt.figure()
    plt.pcolormesh(t,eV,TRPES,cmap='nipy_spectral')
    plt.show()
    
    #fitting it
    test_fit=fits_bidir_2d()
    #starting values
    values_to_fit=[0.9,0.3,4.3]
    values_to_fit_what=['tau1_min','tau1_pos','tau2_pos']
    function_pos=test_fit.interm_fit.bi_exp_decay
    function_min=test_fit.interm_fit.mono_exp_decay
    #make all the sigmas,t,y and add them
    values_interm=[]
    values_interm_what=[]
    tfit=[]
    yfit=[]
    for n in range(TRPES.shape[0]):
        tfit.append(t)
        yfit.append(TRPES[n,:])
        for i in ['sigma_1_min','sigma_1_pos','sigma_2_pos']:
            values_interm.append(max(TRPES[n,:]))
            values_interm_what.append(i)
        values_interm.append(0.)
        values_interm_what.append('moy')
    test_fit.ptot_function(values_interm,values_interm_what)
    test_fit.ptot_function(values_to_fit,values_to_fit_what)
    #plotting the starting values
    '''
    DAS_ini=[test_fit.sigma_1[0],test_fit.sigma_2[0],test_fit.sigma_1[1]]
    pops_ini=[test_fit.mono_exp_decay(t,0,0),test_fit.bi_exp_decay_population(t,0,0),test_fit.mono_exp_decay(-t,1,0)]
    TRPES_ini=np.zeros(ones.shape)
    for n,das in enumerate(DAS_ini):
        #pops_fit[n]=pops_fit[n]/max(pops_fit[n])
        TRPES_ini+=np.reshape(das,(eV.shape[0],1))*ones*pops_ini[n]
    plt.figure()
    plt.pcolormesh(t,eV,TRPES_ini,cmap='nipy_spectral')
    plt.show()
    '''
    #fitting it!
    plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_fit,
                  args=(values_to_fit_what,function_pos,function_min,tfit,yfit),full_output=True)

    #comparing with original
    #remaking the TRPES
    DAS_fit=[test_fit.sigma_1[0],test_fit.sigma_2[0],test_fit.sigma_1[1]]
    DAS=[gauss_pos_tau1,gauss_pos_tau2,gauss_min_tau1]
    #plotting the DAS and comparing with the original
    plt.figure()
    plt.subplot(2,1,1)
    for i in [0,1,2]:
        plt.plot(eV,DAS[i],label=str(i))
    plt.legend()
    plt.subplot(2,1,2)
    for i in [0,1,2]:
        plt.plot(eV,DAS_fit[i],label=str(i))
    plt.legend()
    plt.show()

    #making the reconstructed TRPES
    pops_fit=[test_fit.mono_exp_decay(t,0),test_fit.bi_exp_decay_population(t,0),test_fit.mono_exp_decay(-t,1)]
    TRPES_fit=np.zeros(ones.shape)
    for n,das in enumerate(DAS_fit):
        #pops_fit[n]=pops_fit[n]/max(pops_fit[n])
        TRPES_fit+=np.reshape(das,(eV.shape[0],1))*ones*pops_fit[n]

    plt.figure()
    plt.subplot(3,1,1)
    plt.pcolormesh(t,eV,TRPES,cmap='nipy_spectral')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.pcolormesh(t,eV,TRPES_fit,cmap='nipy_spectral')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.pcolormesh(t,eV,TRPES_fit-TRPES,cmap='nipy_spectral')
    plt.colorbar()
    plt.show()







    
  
