import numpy as np
import scipy.special as sp_erf


class skewed_gauss():
    def __init__(self,center,width, assymetry,intensity,
                 mu_fitted=False,fwhm_fitted=False,alpha_fitted=True,
                 intensity_fitted=False):
        self.mu=center
        self.mu_fitted=mu_fitted
        self.fwhm=width
        self.fwhm_fitted=fwhm_fitted
        self.alpha=assymetry
        self.alpha_fitted=alpha_fitted
        self.intensity=intensity
        self.intensity_fitted=intensity_fitted
        
    def print_gauss(self):
        text='this is an gauss at '+str(self.center)+ 'with fwhm='+str(self.fwhm)+'with alpha='+str(self.alpha)+'and intensity='+str(self.intensity)
        print(text)
        
    def calculate_gauss(self,t):
        '''
        t is an array of values, ascending

        definition of a skewed error-function(see wikiperdia)
        f(x)=2*normal_distrib*PHI(alpha*x)
        normal_distrib:
        g(x)=(1/sigma*sqrt(2pi))*exp(-0.5*((x-mu)/sigma)**2)
        FWHM=2*np.sqrt(2ln2)*sigma
        sigma=FWHM/(2*np.sqrt(2*ln2))
        
        PHI(alpha*x)=1/2*[1+erf(x*alpha/sqrt(2))]
        '''
        sigma=self.fwhm/(2*np.sqrt(2*np.log(2)))
        norm_distr=(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((t-self.mu)/sigma)**2)
        phi=0.5*(1+sp_erf.erf((t-self.mu)*self.alpha/np.sqrt(2)))
        skewed=2*norm_distr*phi
        if max(skewed)!=0.:
            skewed=skewed/max(skewed)
        #print('alpha',self.alpha)
        return self.intensity*skewed
        
if __name__=='__main__':
    print('testing this class')
    t=np.arange(0,5,0.001)
    mu=1
    fwhm=0.5
    alpha=-0.9
    intensity=1
    test_gauss=skewed_gauss(mu,fwhm,alpha,intensity)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t/10,test_gauss.calculate_gauss(t))
    plt.show()











    


        
        
