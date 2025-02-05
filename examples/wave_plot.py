# -*- coding: utf-8 -*-
import autograd.numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
import capytaine as cpy
import mhkit.wave.resource as mhkit
from mhkit.wave.io import ndbc
import pandas as pd
import string
from sklearn.cluster import KMeans
from datetime import datetime
from scipy.optimize import brute
import wecopttool as wot

#plot wave in time domain
from wecopttool import time_results
from xarray import DataArray
from wecopttool import time


# frequency vector
nfreq = 30
f1 = 0.9/nfreq
freq = wot.frequency(f1, nfreq, False)
omega = freq * 2*np.pi

t_dat = time(f1, nfreq, nsubsteps=5)
time_vec_wave= DataArray(data=t_dat, name='time', dims='time', coords=[t_dat])


# regular wave
amplitude = 0.0625
wavefreq = 0.3
waves_reg_fdom = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude,phase=0.0,direction=9.0)
waves_reg_tdom=time_results(waves_reg_fdom, time_vec_wave)
plt.figure() #f-domain
waves_reg_fdom.sel(realization=0).pipe(abs).plot(x='freq')
plt.show(block=True)
plt.figure() #t-domain
waves_reg_tdom.sel(realization=0).plot(linestyle='--')
plt.show(block=True)

# irregular wave
Te = 7.62
gamma = 3.3
fp = 1 / mhkit.energy_period_to_peak_period(Te, gamma)
hs = 1.5
spectrum = lambda f: wot.waves.jonswap_spectrum(f, fp, hs, gamma)
efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "JONSWAP")
waves_irreg_fdom = wot.waves.long_crested_wave(efth,nrealizations=2)
waves_irreg_tdom=time_results(waves_irreg_fdom, time_vec_wave)
plt.figure() #f-domain
waves_irreg_fdom.sel(realization=0).pipe(abs).plot(x='freq')
plt.show(block=True)
plt.figure() #t-domain
waves_irreg_tdom.sel(realization=0).plot(linestyle='-')
waves_irreg_tdom.sel(realization=1).plot(linestyle='--')
plt.show(block=True)