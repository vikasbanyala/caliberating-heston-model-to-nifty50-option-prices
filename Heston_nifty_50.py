#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

    return exp1*term2*exp2
    
    
def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator


def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi=umax/N #dphi is width

    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)


def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    real_integral, err = np.real( quad(integrand, 0, 100, args=args) )

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi


yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yeilds = np.array([6.8,6.8,7.0,7.11,7.11,7.06,7.01,7.15,7.22,7.23,7.43,7.46]).astype(float)/100

curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds)

curve_fit

strike = [21800, 21850, 21900, 21950, 22000]

prices_21800 = [143, 226, 303.8, 351.95, 484.15, 696]
prices_21850 = [119.95, 197, 251.65, 330.1, 473.95, 665.4]
prices_21900 = [92.25, 165.05, 244, 310.76, 431.4, 621]
prices_21950 = [82.5, 153.8, 214.9, 280.6, 400, 595]
prices_22000 = [67.20, 131.95, 204.60, 251.05, 375, 564.7]

prices = [prices_21800, prices_21850, prices_21900, prices_21950, prices_22000]

matu = ['18-Jan-2024', '25-Jan-2024', '01-Feb-2024', '08-Feb-2024', '29-Feb-2024', '28-Mar-2024']

maturities = []
for date in matu:
    maturities.append((dt.strptime(date, '%d-%b-%Y') - dt.today()).days / 365)
    
price_arr = np.transpose(np.array(prices, dtype=object))
np.shape(price_arr)

volSurface = pd.DataFrame(price_arr, index = maturities, columns = strike)

volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
volSurfaceLong.columns = ['maturity', 'strike', 'price']

volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)

# Define variables to be used in optimization
S0 = 21710.8 #as on 05-jan-2024
r = volSurfaceLong['rate'].to_numpy('float')
K = volSurfaceLong['strike'].to_numpy('float')
tau = volSurfaceLong['maturity'].to_numpy('float')
P = volSurfaceLong['price'].to_numpy('float')

params = {"v0": {"x0": 0.1, "lbub": []},
          "kappa": {"x0": 3, "lbub": []},
          "theta": {"x0": 0.05, "lbub": []},
          "sigma": {"x0": 0.3, "lbub": []},
          "rho": {"x0": -0.8, "lbub": []},
          "lambd": {"x0": 0.03, "lbub": []},
          }

x0 = [param["x0"] for key, param in params.items()]
bnds = [param["lbub"] for key, param in params.items()]

def SqErr(x):
    v0, kappa, theta, sigma, rho, lambd = [param for param in x]


    # use rectangular integration function
    err = np.sum( (P-heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))**2 /len(P) )

    # Zero penalty term - no good guesses for parameters
    pen = 0 #np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] )

    return err + pen

result = minimize(SqErr, x0, tol = 1e-3, method='SLSQP', options={'maxiter': 1e4 }, bounds=bnds)

v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
v0, kappa, theta, sigma, rho, lambd

heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
volSurfaceLong['heston_price'] = heston_prices

volSurfaceLong





# In[ ]:




