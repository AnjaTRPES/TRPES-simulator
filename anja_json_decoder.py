import json

from skewed_gauss import skewed_gauss


def encode_anja(z):
    if isinstance(z, skewed_gauss):
        return {"__skewed_gauss__":True,'mu':z.mu, 'mu_fitted':z.mu_fitted,
                'fwhm':z.fwhm,'fwhm_fitted':z.fwhm_fitted,
                'alpha':z.alpha,'alpha_fitted':z.alpha_fitted,
                'intensity':z.intensity,'intensity_fitted':z.intensity_fitted}
    else:
        type_name = z.__class__.__name__
        raise TypeError(f"Object of type '{type_name}' is not JSON serializable")
def decode_anja(dct):
    if "__skewed_gauss__" in dct:
        return skewed_gauss(dct["mu"], dct['fwhm'],dct['alpha'],dct['intensity'],
                            dct["mu_fitted"],
                            dct['fwhm_fitted'],
                            dct['alpha_fitted'],
                            dct['intensity_fitted'])
    return dct

