import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

#testing an offset function I derived
#all times in fs

class fits_2nd(object):
    fwhm=[80.]
    time_offset=[0.]
    sigma_offset=[1]
    tau1=120
    tau2=2400
    tau3=3600
    sigma_1=[]
    sigma_2=[]
    sigma_3=[]
    moy=[]
    counter=0
    

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
        return self.sigma_offset[n]*sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))

    def special_erf(self,t,t1,FWHM):
        sigma=self.fwhm_to_sigma(FWHM)
        lam=self.half_time_to_lambda(t1)
        try:
            g=np.nan_to_num(np.exp(0.5*(lam*sigma)**2-lam*t)*sp.special.erfc((-t+lam*sigma**2)/(np.sqrt(2)*sigma)))
        except FloatingPointError:
            #print('encountered floating point error')
            g=np.zeros(t.shape)
        return g
    
    def mono_exp_decay(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_1[n]*self.special_erf(t,self.tau1,fwhm)

    def mono_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        return self.sigma_offset[n]*((sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm))))-self.mono_exp_decay(t,n)/self.sigma_1[n])
    
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
        return self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))
   
    def bi_exp_decay(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        lam1=self.half_time_to_lambda(self.tau1)
        lam2=self.half_time_to_lambda(self.tau2)
        return self.mono_exp_decay(t,n,several_fwhms)+self.sigma_2[n]*(lam1/(lam2-lam1))*(self.special_erf(t,self.tau1,fwhm)-self.special_erf(t,self.tau2,fwhm))

    def bi_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,several_fwhms)/self.sigma_1[n]
        c=self.bi_exp_decay_population(t,n,several_fwhms)/self.sigma_2[n]
        return self.sigma_offset[n]*(a-b-c)
    
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
        return self.sigma_3[n]*lam1*lam2*(b1+b3)  
    
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
        b=self.sigma_3[n]*lam1*lam2*(b1+b3)       
        return a+b
    def tri_exp_decay_final_state_pop(self,t,n,several_fwhms=False):
        if several_fwhms==False:
            fwhm=self.fwhm[0]
        else:
            fwhm=self.fwhm[n]
        a=sp.special.erfc((-t)/(np.sqrt(2)*self.fwhm_to_sigma(fwhm)))
        b=self.mono_exp_decay(t,n,several_fwhms)/self.sigma_1[n]
        c=self.bi_exp_decay_population(t,n,several_fwhms)/self.sigma_2[n]
        d=self.tri_exp_decay_population(t,n,several_fwhms)/self.sigma_3[n]
        return self.sigma_offset[n]*(a-b-c-d)

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
        b=self.sigma_3[n]*lam1*lam2*(b1+b3)       
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

    
    def fit_function(self,Ptot,what_Ptots,function,tfit,Yfit,floating_t0=False,sigmas=None,several_fwhms=False):
        '''
        Fit function (leastsq fit) with which the exp. values are fitted.
        Ptot: values to fit
        what_Ptots: In string form, to attribute it to the class variables
        function: Fitfunction to use
        tfit: one exp times
        Yfit: one exp. Signals
        sigmas: optional: one sigmas
        '''
        #print 'fitting %i functions',len(tfit)
        self.ptot_function(Ptot,what_Ptots)
        return_roots=[]
        for variable in [self.time_offset,self.sigma_offset,self.sigma_1,self.sigma_2,self.sigma_3,tfit,Yfit,self.fwhm]:
            self.check_whether_list(variable)
        for n in range(len(tfit)):
            if floating_t0==False:
                t=tfit[n]
            else:
                t=tfit[n]+self.time_offset[n]
            if type(sigmas)==type(None):
                return_roots.append(np.sqrt(np.power(function(t,n,several_fwhms)+self.moy[n]-Yfit[n],2)))
            else:
                return_roots.append(np.sqrt(np.power(np.divide(function(t,n,several_fwhms)+self.moy[n]-Yfit[n],sigmas),2)))
        if len(return_roots)==1:
            return return_roots[0]
        else:
            start=return_roots[0]
            #print 'concatenating', len(return_roots)
            for n in range(1,len(return_roots)):
                start=np.concatenate((start,return_roots[n]))            
            return start
    
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
        return self.sigma_1[n]*1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t)**2)/(2*(sigma)**2))
    def fct_auto_corr_with_offset(self,t,n,several_fwhms=False):
        sigma=self.fwhm_to_sigma(self.fwhm[n])
        return self.sigma_1[n]*1/(sigma*np.sqrt(2*np.pi))*np.exp(-((t)**2)/(2*(sigma)**2))+self.offset(t,n)
     
    def extract_Deltas_plsqs(self,plsq,cov,whatPtots,function,tfit,Yfit):
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
            if cov.any()==None:
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




"""
#load an example:
data=np.loadtxt('2- benzyl_PE.dat')
tfit=data[:,0]
yfit=data[:,1]

test_fit=fits()

tau1=360.
tau2=3600
test_fit.tau2=3600.
test_fit.tau3=4300
fwhm=80.
offset=300.
sigma_1=data[(np.abs(data[:,0]-tau1)).argmin(),1]*np.pi
sigma_2=data[(np.abs(data[:,0]-test_fit.tau2)).argmin(),1]*np.pi
test_fit.sigma_3=[data[(np.abs(data[:,0]-test_fit.tau3)).argmin(),1]*np.pi]
fixed_constants=[data[0:4,1].mean(),80.]
whats_fixed=['moy','fwhm']
test_fit.moy=[data[0:4,1].mean()]
values_to_be_fitted=[tau1,tau2,sigma_1,sigma_2,offset,fwhm]
values_to_be_fitted_what=['tau1','tau2','sigma_1','sigma_2','time_offset','fwhm']
values_needed_to_be_added_to_delta=values_to_be_fitted_what[:]
values_to_be_fitted,values_to_be_fitted_what=compare_fit_fixed_constants(fixed_constants,whats_fixed,values_to_be_fitted,values_to_be_fitted_what)
if values_to_be_fitted!=[]:
    #set the fixed values:
    test_fit.ptot_function(fixed_constants,whats_fixed)
    test_fit.ptot_function(values_to_be_fitted,values_to_be_fitted_what)
    print 'just bevore fitting'
    #now fit it
    plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,values_to_be_fitted,
                  args=(values_to_be_fitted_what,test_fit.bi_exp_decay,[tfit],[yfit]),full_output=True)
    #get the incertitudes
    plsqs,Deltas=test_fit.extract_Deltas_plsqs(plsq,cov,values_to_be_fitted_what,test_fit.bi_exp_decay,[data[:,0]],[data[:,1]])
    #make the new object, structure of data[Xexp,Yexp,total_fit, tau1,autocorr]
    
    plsqs=[test_fit.tau1,test_fit.sigma_1[0],test_fit.time_offset[0],test_fit.fwhm]
    Deltas=[]
    g=0
    for n in range(len(values_needed_to_be_added_to_delta)):
        if values_needed_to_be_added_to_delta[n] in values_to_be_fitted:
            Deltas.append(Deltas2[n-g])
        else:
            Deltas.append(None)
            g=g+1
    
    data_new=np.ones((data.shape[0],5))
    data_new[:,0:2]=data
    data_new[:,2]=test_fit.bi_exp_decay(tfit-test_fit.time_offset[0],0)
    data_new[:,3]=test_fit.tri_exp_decay(data[:,0]-test_fit.time_offset[0],0)
    data_new[:,4]=test_fit.fct_auto_corr(data[:,0]-test_fit.time_offset[0],0)
    print 'test_fit_values'
    print'now here'
    print '----------------------------'
    print 'plsqs',plsqs
    print 'Deltas',Deltas
    print '----------------------'

plt.figure()
plt.plot(data_new[:,0],data_new[:,1],'ko')
plt.plot(data_new[:,0],data_new[:,2],'r',label='fit method1')
plt.plot(data_new[:,0],test_fit.bi_exp_decay_final_state_pop(data_new[:,0],0),label='final state')
#plt.plot(tfit+test_fit.time_offset[0],test_fit.mono_exp_decay(tfit,0),'b')
plt.show()

#now direct:
test_fit2=fits()
test_fit2.moy=[data[0:4,1].mean()]
Ptot=[fwhm,tau1,offset,sigma_1]
what_Ptots=['fwhm','tau1','time_offset','sigma_1']
test_fit2.ptot_function(Ptot,what_Ptots)

plt.plot(tfit,yfit,'ko',label='exp.points')
plsq,cov,info,msg,ier=leastsq(test_fit2.fit_function,Ptot,
                              args=(what_Ptots,test_fit2.mono_exp_decay,[tfit],[yfit]),full_output=True)
plt.plot(tfit,test_fit2.mono_exp_decay(tfit-test_fit2.time_offset[0],0),label='after fit')
plt.legend()
plt.show()
plsqs,Deltas=test_fit.extract_Deltas_plsqs(plsq,cov,what_Ptots,test_fit2.mono_exp_decay,[tfit],[yfit])

print 'test_fit_values'
print'now here'
print '----------------------------'
print 'plsqs',plsqs
print 'Deltas',Deltas
print '----------------------'

"""



"""
#try if my concept works for fitting the same decay two times:
plt.figure()
tfits=[tfit,tfit]
yfits=[yfit,yfit]

#get the starting variables
fwhm=80.
tau1=300.
tau2=3600.
time_offset=300.

#get the starting values for the sigmas
sigma_1=yfit[(np.abs(tfit-time_offset-tau1)).argmin()]*np.pi
sigma_2=yfit[(np.abs(tfit-time_offset-tau2)).argmin()]*np.pi

Ptot=[fwhm,tau1,tau2,
      time_offset,time_offset,
      sigma_1,sigma_1,
      sigma_2,sigma_2]
what_Ptots=['fwhm','tau1','tau2',
            'time_offset','time_offset',
            'sigma_1','sigma_1',
            'sigma_2','sigma_2']
plsq,cov,info,msg,ier=leastsq(test_fit.fit_function,Ptot,
                              args=(what_Ptots,test_fit.bi_exp_decay,tfits,yfits),full_output=True)
print plsq
print cov

plt.plot(tfit,yfit,'ko',label='exp.points')                        
plt.plot(tfits[0]+test_fit.time_offset[0],test_fit.bi_exp_decay(tfits[0],0),label='after fit0')
plt.plot(tfits[1]+test_fit.time_offset[1],test_fit.bi_exp_decay(tfits[1],1),label='after fit1')
plt.legend()
plt.show()


#things to do: include an 'moy'
#make subfunction to extract plsq, cov
#integrate this class in my pyqt script -> much 'leaner code'
"""







