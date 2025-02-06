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
from mhkit.wave.io import cdip
import matplotlib.pyplot as plt
import os

current_dir = os.getcwd()
results_folder = os.path.join(current_dir, "results") # Define the path to the "results" folder
if not os.path.exists(results_folder): # Step 3: Create the "results" folder if it doesn't exist
    os.makedirs(results_folder)


#------------------------------Select Buoy Number to get wave data---------------------
station_number = "243" #"243" for JP
#start_date = "2020-04-01"
#end_date = "2020-04-30"
#parameters = ["waveHs", "waveTp","waveTe"] #, "waveMeanDirection"]
data = cdip.request_parse_workflow(
    station_number=station_number)
    #parameters=parameters) #,
    #start_date=start_date,
    #end_date=end_date,
#)
print("\n")
print(f"Returned data: {data['data']['wave'].keys()} \n")
Hm0=data['data']['wave']['waveHs'].values
Tp=data['data']['wave']['waveTp'].values
# Generate 2% noise
noise_percentage = 0.02  # 2% noise
noise = Tp * noise_percentage * np.random.randn(len(Tp))
# Add noise to the original data
Tp = Tp + noise
data['data']['wave']['waveTp'] = Tp
gamma = 3.3 #float or int #Peak enhancement factor for JONSWAP spectrum
Te=mhkit.peak_period_to_energy_period(Tp, gamma) # as It did not have Te, we compute it from Tp
numData=len(Te)

data_wave = pd.DataFrame({
    "Hm0": Hm0,
    "Te": Te
})

# clusters
N = 10
sea_states_labels = list(string.ascii_uppercase[0:N])
raster_order = -10
km = KMeans(n_clusters=N, random_state=1).fit(data_wave[["Hm0", "Te"]])
weights = [(km.labels_ == i).sum() / len(km.labels_) for i in range(N)]
sea_states = pd.DataFrame(km.cluster_centers_, columns=["Hm0", "Te"])
rho = 1025
g = 9.81
sea_states["power"] =  (rho*g**2)/(64*np.pi)*(sea_states.Hm0**2)*sea_states.Te / 1000 #kW/m #wave power per unit width of wave crest
sea_states["weight"] = weights
sea_states.sort_values("Hm0", inplace=True, ascending=True)
idx = sea_states.index
idx = [int(np.where(idx == i)[0]) for i in np.arange(N)]
idx = [idx[i] for i in km.labels_]
sea_states.reset_index(drop=True, inplace=True)
P_density_average=sum(sea_states.weight*sea_states.power)
print("Average annual power density [kW]:", P_density_average)  # Average annual power density


# representative sea state spectra (JONSWAP)
nfreq = 127
f1 = 0.6/nfreq #0.42/nfreq
waves = []
spectra = []
fp_vec=[]
for i, sea_state in sea_states.iterrows():
    fp = 1 / mhkit.energy_period_to_peak_period(sea_state.Te, gamma)
    Hm0 = sea_state.Hm0
    spectrum = lambda f: wot.waves.jonswap_spectrum(f, fp, Hm0, gamma)
    efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "JONSWAP")
    wave = wot.waves.long_crested_wave(efth,nrealizations=1)
    wave.attrs['weight'] = sea_state.weight
    waves.append(wave)
    spectra.append(efth)
    fp_vec.append(fp)

#Figure 5
cmap_qualitative = cm.tab10
fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.scatter(data_wave.Te, data_wave.Hm0, c=idx, s=4, cmap=cmap_qualitative, rasterized=True,alpha=1.0)
ax.scatter(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0], s=40, marker="x", color="w")
for x, y, lbl in zip(sea_states["Te"], sea_states.Hm0, sea_states_labels):
    plt.text(x + 0.1, y + 0.1, lbl, fontsize=18, color='k')
ax.set_xlabel("Energy period, $T_e$ [s]",fontsize=17)
ax.set_ylabel("Significant wave height, " + "$H_{m0}$ [m]",fontsize=17)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title("Jennette's Pier Wave Data",fontsize=17)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(results_folder, "JT_Pier_Wave_Data.pdf"), format='pdf', dpi=300, bbox_inches='tight')
plt.show(block=True)


#Figure 6
fig, ax = plt.subplots(1,1, figsize=(6,6))
for i in range(len(waves))[::-1]:
    wave = waves[i]
    spectrum = spectra[i]
    f = wave.omega/(2*np.pi)
    ax.plot([f[0], f[-1]], [0, 0], "k-")
    ax.plot(f, spectrum, '-', color=cmap_qualitative.colors[i], marker='.')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    x = f[np.where(spectrum[:,0] == max(spectrum[:,0]))].values
    y = max(spectrum.values)
    plt.text(x, y, sea_states_labels[i])
    plt.xlabel('Frequency [Hz]',fontsize=17)
    plt.ylabel('Spectrum, $S$ [m$^2$/Hz]',fontsize=17)
    ax.set_title("Jennette's Pier Wave Distributions",fontsize=17)
    plt.xlim([0.0, 0.3])
    ax.tick_params(axis='both', which='major', labelsize=15)
    #plt.ylim([0, 100])
plt.savefig(os.path.join(results_folder, "JT_Pier_Wave_Dist.pdf"), format='pdf', dpi=300, bbox_inches='tight')
plt.show(block=True)

#plot wave in freq domain
plt.figure(figsize=(10, 6))
for i, wave in enumerate(waves):
    wave.sel(realization=0).pipe(abs).plot(x='freq', label=f"Wave {i}", add_legend=False)
plt.legend(title="Waves")  # Add legend to distinguish different waves
#plt.title("All Waves on the Same Plot")
plt.xlabel("Frequency (Hz)")  # Replace with the appropriate frequency unit
#plt.ylabel()  # Replace with the appropriate label for the y-axis
plt.show(block=True)

#plot wave in time domain
from wecopttool import time_results
from xarray import DataArray
from wecopttool import time
t_dat = time(f1, nfreq, nsubsteps=5)
time_vec_wave= DataArray(data=t_dat, name='time', dims='time', coords=[t_dat])
waves_tdom = [time_results(wave, time_vec_wave) for wave in waves]
fig, ax = plt.subplots(1,1, figsize=(6,6))
for i, wave in enumerate(waves_tdom):
    wave.sel(realization=0).plot(label=f" {sea_states_labels[i]}", add_legend=False)
plt.legend(title="Waves",ncol=2,fontsize=12)  # Add legend to distinguish different waves
#plt.title("All Waves on the Same Plot")
plt.xlabel("Time [s]",fontsize=17)  # Replace with the appropriate frequency unit
plt.ylabel("Wave Amp [m]",fontsize=17)  # Replace with the appropriate label for the y-axis
plt.xlim([0.0, 20.0]) #limit to 50 s
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_title("Jennette's Pier Wave Time Series",fontsize=17)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(results_folder, "JT_Pier_Wave_Time_series.pdf"), format='pdf', dpi=300, bbox_inches='tight')
plt.show(block=True)
k=1

