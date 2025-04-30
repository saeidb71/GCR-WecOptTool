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
import os
from netCDF4 import Dataset
from mhkit.wave.io import cdip
import matplotlib.pyplot as plt
from wecopttool import time_results
from xarray import DataArray
from wecopttool import time
import matplotlib.dates as mdates
import pickle as pkl
from  utilities import calculate_power_flows
from utilities import plot_power_flow

class wecOpt:
    def __init__(self,saved_rslts_addrs):
        """Constructor method to initialize attributes"""
        self.gamma = 3.3 #float or int #Peak enhancement factor for JONSWAP spectrum
        self.rho = 1025
        self.g = 9.81
        self.fmax=0.9 #0.9 # Hz #0.6
        self.nfreq = 50 #127# 30 #127
        self.f1 = self.fmax/self.nfreq #0.42/nfreq
        self.saved_rslts_addrs=saved_rslts_addrs
        self.nsubsteps = 5
        self.r1 = 0.88
        self.r2_0 = 0.35
        self.h2_0 = 0.37
        self.V0_disp = 1/3*np.pi*self.h2_0*(self.r1**2+self.r2_0**2+(self.r1*self.r2_0))+np.pi*self.r1**2*0.17
        #fb.volume=~self.V0_disp

        current_dir = os.getcwd()
        results_folder = os.path.join(current_dir, saved_rslts_addrs) # Define the path to the "results" folder
        if not os.path.exists(results_folder): # Step 3: Create the "results" folder if it doesn't exist
            os.makedirs(results_folder)
        self.results_folder=results_folder


    def load_Wave_Data(self,station_number):
        gamma=self.gamma
        rho=self.rho
        g=self.g
        file_name_wec = f"Wave_Data/{station_number}p1_historic.nc"
        nc = Dataset(file_name_wec, mode='r')
        #print(nc.variables.keys())
        if station_number=="253": 
            start_date = "2021-4-18" 
            end_date =  "2021-12-11" 
        elif station_number=="269":
            start_date= "2024-05-15"
            end_date =  "2024-11-03"
        elif station_number=="243":
            start_date= "2022-01-1"
            end_date =  "2023-01-1"
        parameters = ["waveHs", "waveTp","waveTe"] #, "waveMeanDirection"]
        data = cdip.request_parse_workflow(
            nc=nc,
            station_number=station_number,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date)   
        
        # add T_e to the data if it does not exist
        Hm0=data['data']['wave']['waveHs'].values
        Tp=data['data']['wave']['waveTp'].values
        if 'waveTe' not in data['data']['wave']:
            Te = mhkit.peak_period_to_energy_period(Tp, gamma)
            data['data']['wave']['waveTe'] = Te
        else:
            Te=data['data']['wave']['waveTe'].values

        # compute power density: kW/m
        data['data']['wave']['wavePwrDnsty'] = (rho * g**2) / (64 * np.pi) * \
            (data['data']['wave']['waveHs']**2) * data['data']['wave']['waveTe'] / 1000
        
        df_wave = data['data']['wave']  # Assuming df is a Pandas DataFrame with a DateTimeIndex
        monthly_data = {} # Create a dictionary where each key is 'YYYY-MM' and each value is the subset DataFrame
        for period, group_df in df_wave.groupby(df_wave.index.to_period('M')):
            key = period.strftime('%Y-%m') # Convert the period (e.g. Period('2023-06', 'M')) to a string like '2023-06'
            monthly_data[key] = group_df
        all_times = df_wave.index # 1) Get them directly as a DatetimeIndex:
        print(all_times)
        all_times_list = df_wave.index.tolist() # 2) Convert them to a Python list of Timestamps:

        data_wave = pd.DataFrame({
            "Hm0": Hm0,
            "Te": Te,
            "Tp": Tp,
        })     

        self.data_wave=data_wave
        self.df_wave=df_wave

    def plot_Wave_Data(self):
        df_wave=self.df_wave
        results_folder=self.results_folder

        plt.figure(figsize=(12, 6))
        plt.plot(df_wave.index, df_wave['wavePwrDnsty'], linestyle='-', label='Wave Power Density')
        mean_value = df_wave['wavePwrDnsty'].mean()
        plt.axhline(y=mean_value, color='r', linestyle='--', 
                    label=f'Mean = {mean_value:.2f} kW/m')
        plt.title("Wave Power Density vs Time")
        plt.xlabel("Time")
        plt.ylabel("Wave Power Density (kW/m)")
        # Format the x-axis to display dates nicely
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        # Adjust layout and save the figure with a tight bounding box using a relative path.
        plt.tight_layout()
        plt.savefig(f'{results_folder}/wave_power_density.png', bbox_inches='tight')
        plt.show()

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        line1, = ax1.plot(df_wave.index, df_wave['waveTe'], color='tab:blue', label='Wave Te',linewidth=1.0)
        ax1.set_ylabel(r'$T_e$', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        line2, = ax2.plot(df_wave.index, df_wave['waveHs'], color='tab:orange', label='Wave Hs',linewidth=1.0)
        ax2.set_ylabel(r'$H_{s}$', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax1.set_xlabel('Time')
        plt.title("Wave Energy Period (waveTe) and Significant Wave Height (waveHs)")
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')
        fig.tight_layout()
        ax1.set_ylim(0, 17)
        ax2.set_ylim(0, 9.5)
        plt.savefig(f'{results_folder}/plot_two_yaxes.png', bbox_inches='tight')
        plt.show()


    def sea_state_clustering(self,N):
        rho=self.rho
        g=self.g
        gamma=self.gamma
        data_wave=self.data_wave
        results_folder=self.results_folder
        f1=self.f1
        nfreq=self.nfreq
        N = 10 # number of clusters
        sea_states_labels = list(string.ascii_uppercase[0:N])
        raster_order = -10
        km = KMeans(n_clusters=N, random_state=1).fit(data_wave[["Hm0", "Te"]])
        weights = [(km.labels_ == i).sum() / len(km.labels_) for i in range(N)]
        sea_states = pd.DataFrame(km.cluster_centers_, columns=["Hm0", "Te"])
        sea_states["power"] =  (rho*g**2)/(64*np.pi)*(sea_states.Hm0**2)*sea_states.Te / 1000
        sea_states["weight"] = weights
        sea_states["Tp"] = mhkit.energy_period_to_peak_period(sea_states.Te, gamma)
        sea_states["lambda_opt"]=g*sea_states["Tp"]**2/2/np.pi
        sea_states["width_opt"]=sea_states["lambda_opt"]/2/np.pi
        sea_states["opt_pwr_kW"]=sea_states["power"]*sea_states["width_opt"]
        sea_states.sort_values("Hm0", inplace=True, ascending=True)
        idx = sea_states.index
        idx = [int(np.where(idx == i)[0]) for i in np.arange(N)]
        idx = [idx[i] for i in km.labels_]
        sea_states.reset_index(drop=True, inplace=True)
        P_density_average=sum(sea_states.weight*sea_states.power)
        P_opt_average=sum(sea_states.weight*sea_states.opt_pwr_kW)
        print(sea_states)
        print("Average annual power density [kW/m]:", P_density_average)  # Average annual power density
        print("opt annual power [kW]:", P_opt_average)  

        with open(f"{results_folder}/sea_state_summary.txt", "w") as file:
            file.write(sea_states.to_string())
            file.write("\n")
            file.write(f"Average annual power density [kW/m]: {P_density_average}\n")
            file.write(f"Average annual power [kW]: {P_opt_average}\n")

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

        self.km=km
        self.sea_states=sea_states
        self.sea_states_labels=sea_states_labels
        self.waves=waves
        self.spectra=spectra
        self.idx=idx

    def plot_Sea_States(self):
        data_wave=self.data_wave
        km=self.km
        sea_states=self.sea_states
        sea_states_labels=self.sea_states_labels
        results_folder=self.results_folder
        waves=self.waves
        spectra=self.spectra
        nfreq=self.nfreq
        f1=self.f1
        idx=self.idx
        nsubsteps=self.nsubsteps

        cmap_qualitative = cm.tab10
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        ax.scatter(data_wave.Te, data_wave.Hm0, c=idx, s=4, cmap=cmap_qualitative, rasterized=True,alpha=1.0)
        ax.scatter(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0], s=40, marker="x", color="w")
        for x, y, lbl in zip(sea_states["Te"], sea_states.Hm0, sea_states_labels):
            plt.text(x + 0.1, y + 0.1, lbl, fontsize=18, color='k')
        ax.set_xlabel("Energy period, $T_e$ [s]",fontsize=17)
        ax.set_ylabel("Significant wave height, " + "$H_{m0}$ [m]",fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title("Beaver Island Wave Data",fontsize=17)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(results_folder, "Beaver_Island_Wave_Data.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        plt.show(block=True)


        #Figure 6
        fig, ax = plt.subplots(1,1, figsize=(7.5,7.5))
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
            ax.set_title("Beaver Island Wave Distributions",fontsize=17)
            plt.xlim([0.03, 0.5])
            ax.tick_params(axis='both', which='major', labelsize=15)
            #plt.ylim([0, 100])
        plt.savefig(os.path.join(results_folder, "Beaver_Island_Wave_Dist.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        plt.show(block=True)

        #plot wave in freq domain
        plt.figure(figsize=(10, 6))
        for i, wave in enumerate(waves):
            wave.sel(realization=0).pipe(abs).plot(x='freq', label=f"Wave {i}", add_legend=False)
        plt.legend(title="Waves")  
        plt.xlabel("Frequency (Hz)") 
        plt.savefig(os.path.join(results_folder, "Beaver_Island_freq_dsit.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        plt.show(block=True)


        t_dat = time(f1, nfreq, nsubsteps=nsubsteps)
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
        ax.set_title("Beaver Island Wave Time Series",fontsize=17)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(results_folder, "Beaver_Island_Wave_Time_series.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        plt.show(block=True)
        k=1

    def compare_Meshes(self):
        nfreq=self.nfreq
        f1=self.f1 
        freq = wot.frequency(f1, nfreq, False) # frequency vector
        omega = freq * 2*np.pi

        mesh_1 = wot.geom.WaveBot().mesh(mesh_size_factor = 0.48) #0.48
        fb_1 = cpy.FloatingBody.from_meshio(mesh_1, name="WaveBot")
        fb_1.add_translation_dof(name="Heave")
        bem_data_1 = wot.run_bem(fb_1, freq)

        mesh_2 = wot.geom.WaveBot().mesh(mesh_size_factor = 0.98) #0.48
        fb_2 = cpy.FloatingBody.from_meshio(mesh_2, name="WaveBot")
        fb_2.add_translation_dof(name="Heave")
        bem_data_2 = wot.run_bem(fb_2, freq)

        # Assume bem_data_1 and bem_data_2 are already loaded

        # Frequency axis (assumed same for both)
        omega = bem_data_1['omega']
        frequency_hz = omega / (2 * np.pi)

        # === Extract and squeeze real-valued variables ===
        added_mass_1 = bem_data_1['added_mass'].squeeze()
        added_mass_2 = bem_data_2['added_mass'].squeeze()

        radiation_damping_1 = bem_data_1['radiation_damping'].squeeze()
        radiation_damping_2 = bem_data_2['radiation_damping'].squeeze()

        # === Extract and squeeze complex-valued variables ===
        diffraction_force_1 = np.abs(bem_data_1['diffraction_force'].squeeze())
        diffraction_force_2 = np.abs(bem_data_2['diffraction_force'].squeeze())

        froude_krylov_1 = np.abs(bem_data_1['Froude_Krylov_force'].squeeze())
        froude_krylov_2 = np.abs(bem_data_2['Froude_Krylov_force'].squeeze())

        excitation_force_1 = np.abs(bem_data_1['excitation_force'].squeeze())
        excitation_force_2 = np.abs(bem_data_2['excitation_force'].squeeze())

        # === Plotting ===
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))

        # Added Mass
        axs[0, 0].plot(frequency_hz, added_mass_1, label='BEM 1')
        axs[0, 0].plot(frequency_hz, added_mass_2, '--', label='BEM 2')
        axs[0, 0].set_title('Added Mass')
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Radiation Damping
        axs[0, 1].plot(frequency_hz, radiation_damping_1, label='BEM 1')
        axs[0, 1].plot(frequency_hz, radiation_damping_2, '--', label='BEM 2')
        axs[0, 1].set_title('Radiation Damping')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Diffraction Force
        axs[1, 0].plot(frequency_hz, diffraction_force_1, label='BEM 1')
        axs[1, 0].plot(frequency_hz, diffraction_force_2, '--', label='BEM 2')
        axs[1, 0].set_title('|Diffraction Force|')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Force (abs)')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Froude-Krylov Force
        axs[1, 1].plot(frequency_hz, froude_krylov_1, label='BEM 1')
        axs[1, 1].plot(frequency_hz, froude_krylov_2, '--', label='BEM 2')
        axs[1, 1].set_title('|Froude-Krylov Force|')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Force (abs)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Excitation Force
        axs[2, 0].plot(frequency_hz, excitation_force_1, label='BEM 1')
        axs[2, 0].plot(frequency_hz, excitation_force_2, '--', label='BEM 2')
        axs[2, 0].set_title('|Excitation Force|')
        axs[2, 0].set_xlabel('Frequency (Hz)')
        axs[2, 0].set_ylabel('Force (abs)')
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Turn off last empty subplot (2,1)
        axs[2, 1].axis('off')
        plt.tight_layout()
        plt.savefig(f"{self.results_folder}/comp_BEM_data.png", dpi=300, bbox_inches='tight')
        plt.show()

        # === Extract values ===
        inertia_1 = bem_data_1['inertia_matrix'].values
        inertia_2 = bem_data_2['inertia_matrix'].values
        stiffness_1 = bem_data_1['hydrostatic_stiffness'].values
        stiffness_2 = bem_data_2['hydrostatic_stiffness'].values

        # === Print comparison ===
        print("\n--- Inertia Matrix Comparison ---")
        print("BEM 1:\n", inertia_1)
        print("BEM 2:\n", inertia_2)
        print("Difference:\n", inertia_2 - inertia_1)

        print("\n--- Hydrostatic Stiffness Comparison ---")
        print("BEM 1:\n", stiffness_1)
        print("BEM 2:\n", stiffness_2)
        print("Difference:\n", stiffness_2 - stiffness_1)

        # === Save to file ===
        save_path = f"{self.results_folder}/BEM_data"
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, "inertia_and_stiffness_comparison.txt")

        with open(f"{self.results_folder}/comp_inertia_and_hydrostatic.txt", 'w') as f:
            f.write("--- Inertia Matrix Comparison ---\n")
            f.write("BEM 1:\n" + str(inertia_1) + "\n")
            f.write("BEM 2:\n" + str(inertia_2) + "\n")
            f.write("Difference:\n" + str(inertia_2 - inertia_1) + "\n\n")
            
            f.write("--- Hydrostatic Stiffness Comparison ---\n")
            f.write("BEM 1:\n" + str(stiffness_1) + "\n")
            f.write("BEM 2:\n" + str(stiffness_2) + "\n")
            f.write("Difference:\n" + str(stiffness_2 - stiffness_1) + "\n")

    def Load_WEC(self):
        results_folder=self.results_folder

        nfreq=self.nfreq
        f1=self.f1 

        mesh = wot.geom.WaveBot().mesh(mesh_size_factor = 0.48) #0.48
        fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")
        fb.add_translation_dof(name="Heave")
        ndof = fb.nb_dofs

        fb.show_matplotlib()
        #plt.gca().view_init(elev=4, azim=120)
        #plt.savefig(f'{results_folder}/wec_shape_1.png')

        """show_normals=True
        fb.show(show_normals)"""

        freq = wot.frequency(f1, nfreq, False) # frequency vector
        omega = freq * 2*np.pi
        bem_data = wot.run_bem(fb, freq) # run Capytaine (BEM)
        wec = wot.WEC.from_bem( # create WEC object
            bem_data,
            constraints=None,
            friction=None,
            f_add=None,
        )
        #fb.volume=~self.V0_disp
        # Convert omega to frequency in Hz
        omega = bem_data['omega']
        frequency_hz = omega / (2 * np.pi)

        # Extract real-valued data
        added_mass = bem_data['added_mass'].squeeze()
        radiation_damping = bem_data['radiation_damping'].squeeze()

        # Extract and compute absolute value of complex data
        diffraction_force = bem_data['diffraction_force'].squeeze()
        froude_krylov = bem_data['Froude_Krylov_force'].squeeze()
        excitation_force = np.abs(bem_data['excitation_force'].squeeze())

        # Compute |diffraction + Froude-Krylov|
        excitation_force_2 = np.abs(diffraction_force + froude_krylov)

        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Subplot 1: Added Mass & Radiation Damping
        axs[0, 0].plot(frequency_hz, added_mass, label='Added Mass')
        axs[0, 0].plot(frequency_hz, radiation_damping, label='Radiation Damping')
        axs[0, 0].set_title('Added Mass & Radiation Damping')
        axs[0, 0].set_xlabel('Frequency (Hz)')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Subplot 2: |Diffraction Force|
        axs[0, 1].plot(frequency_hz, np.abs(diffraction_force))
        axs[0, 1].set_title('|Diffraction Force|')
        axs[0, 1].set_xlabel('Frequency (Hz)')
        axs[0, 1].set_ylabel('Force (abs)')
        axs[0, 1].grid(True)

        # Subplot 3: |Froude-Krylov Force|
        axs[1, 0].plot(frequency_hz, np.abs(froude_krylov))
        axs[1, 0].set_title('|Froude-Krylov Force|')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Force (abs)')
        axs[1, 0].grid(True)

        # Subplot 4: |Excitation Force| and |Diffraction + FK|
        axs[1, 1].plot(frequency_hz, excitation_force, label='|Excitation Force|')
        axs[1, 1].plot(frequency_hz, excitation_force_2, '--', label='|Diffraction + Froude-Krylov|')
        axs[1, 1].set_title('Excitation Forces')
        axs[1, 1].set_xlabel('Frequency (Hz)')
        axs[1, 1].set_ylabel('Force (abs)')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.results_folder}/BEM_data.png", dpi=300, bbox_inches='tight')
        plt.show()
        inertia_matrix = bem_data['inertia_matrix'].values
        hydrostatic_stiffness = bem_data['hydrostatic_stiffness'].values
        # Print Inertia Matrix and Hydrostatic Stiffness
        print("\nInertia Matrix:")
        print(bem_data['inertia_matrix'].values)

        print("\nHydrostatic Stiffness:")
        print(bem_data['hydrostatic_stiffness'].values)


        # Write to file
        with open(f"{self.results_folder}/inertia_and_hydrostatic.txt", 'w') as f:
            f.write("Inertia Matrix:\n")
            f.write(str(inertia_matrix) + "\n\n")
            f.write("Hydrostatic Stiffness:\n")
            f.write(str(hydrostatic_stiffness) + "\n")


        hydro_data=wot.add_linear_friction(bem_data=bem_data)
        #hydro_data = wot.linear_hydrodynamics(bem_data, mass, stiffness)
        Zi = wot.hydrodynamic_impedance(hydro_data).sel(influenced_dof='Heave', radiating_dof='Heave')


        self.omega=omega
        self.fb=fb
        self.wec=wec
        self.bem_data=bem_data
        self.freq=freq
        self.hydro_data=hydro_data
        self.Zi=Zi

    def pto_impedance(self,drivetrain_inertia=2.0, drivetrain_stiffness = 0.0):

        omega=self.omega
        gear_ratio = 12.0
        torque_constant = 6.7
        winding_resistance = 0.5
        winding_inductance = 0.0
        drivetrain_friction = 1.0

        drivetrain_impedance = (1j*omega*drivetrain_inertia +
                                drivetrain_friction + -1j/(omega)*drivetrain_stiffness)

        winding_impedance = winding_resistance + 1j*omega*winding_inductance

        pto_impedance_11 = -1* gear_ratio**2 * drivetrain_impedance
        off_diag = np.sqrt(3.0/2.0) * torque_constant * gear_ratio
        pto_impedance_12 = -1*(off_diag+0j) * np.ones(omega.shape)
        pto_impedance_21 = -1*(off_diag+0j) * np.ones(omega.shape)
        pto_impedance_22 = winding_impedance
        pto_impedance_mat = np.array([[pto_impedance_11, pto_impedance_12],
                                [pto_impedance_21, pto_impedance_22]])
        return pto_impedance_mat
    
    def reg_wave(self,amplitude,wavefreq):
        results_folder=self.results_folder
        nfreq=self.nfreq
        f1=self.f1 
        waves_reg_fdom = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude)
        self.waves_reg_fdom=waves_reg_fdom
        nsubsteps=self.nsubsteps
        
        plt.figure()
        waves_reg_fdom.pipe(abs).plot()
        plt.savefig(f'{results_folder}/wave_reg_fdom.png')

        t_dat = time(f1, nfreq)
        time_vec_wave= DataArray(data=t_dat, name='time', dims='time', coords=[t_dat])
        waves_reg_tdom=time_results(waves_reg_fdom, time_vec_wave)
        plt.figure()
        waves_reg_tdom.plot()
        plt.savefig(f'{results_folder}/wave_reg_tdom.png')

        self.waves_reg_fdom=waves_reg_fdom
        self.wavefreq_reg =  wavefreq
        self.amplitude_reg = amplitude

    def irreg_wave(self,Te,hs):
        results_folder=self.results_folder
        gamma = self.gamma
        nfreq=self.nfreq
        nsubsteps=self.nsubsteps
        f1=self.f1 
        fp = 1 / mhkit.energy_period_to_peak_period(Te, gamma)
        spectrum = lambda f: wot.waves.jonswap_spectrum(f, fp, hs, gamma)
        efth = wot.waves.omnidirectional_spectrum(f1, nfreq, spectrum, "JONSWAP")
        waves_irreg_fdom = wot.waves.long_crested_wave(efth,nrealizations=2)
        self.waves_irreg_fdom=waves_irreg_fdom

        t_dat = time(f1, nfreq, nsubsteps=nsubsteps)
        time_vec_wave= DataArray(data=t_dat, name='time', dims='time', coords=[t_dat])
        waves_irreg_tdom=time_results(waves_irreg_fdom, time_vec_wave)
        self.waves_irreg_tdom=waves_irreg_tdom

        plt.figure()
        waves_irreg_fdom.sel(realization=0).pipe(abs).plot()
        waves_irreg_fdom.sel(realization=1).pipe(abs).plot(linestyle='--')
        plt.savefig(f'{results_folder}/wave_irreg_fdom.png')

        plt.figure()
        waves_irreg_tdom.sel(realization=0).plot()
        waves_irreg_tdom.sel(realization=1).plot(linestyle='--')
        plt.savefig(f'{results_folder}/wave_irreg_tdom.png')

        self.waves_irreg_tdom=waves_irreg_tdom
        self.H_sig=hs       # Significant wave height for irregular waves
        self.T_dom=Te  
        self.Tp_irreg=1/fp

    def verification(self,controller, scale_x_opt, nstate_opt, waves,flg_obj):
        fb=self.fb
        wec=self.wec
        nsubsteps=self.nsubsteps
        # PTO
        pto = wot.pto.PTO(
            fb.nb_dofs,
            np.eye(fb.nb_dofs),
            controller,
            self.pto_impedance(), #None, #pto_impedance(), #None
            None,
            ["PTO_Heave"],
        )
        # WEC additional forces
        wec.forces['PTO'] = pto.force_on_wec
        """def const_f_pto(wec, x_wec, x_opt, waves):
            #f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
            power=    pto.mechanical_power(wec, x_wec, x_opt, waves, nsubsteps)
            return 0.4 - np.abs(power.flatten()) # power alwasy be nagative: generator (not motor)
        wec.constraints = [{'type': 'ineq',
                            'fun': const_f_pto,
                            }]"""
        # objective function
        if flg_obj=="elec":
            objective = pto.average_power# pto.average_power #pto.mechanical_average_power # pto.average_power
        elif flg_obj=="mech":
            objective = pto.mechanical_average_power 

        # optimal controller
        results = wec.solve(waves, objective, nstate_opt, scale_x_opt=scale_x_opt, optim_options={"disp": True,"maxiter":200})
        # post-process
        wec_fdom, wec_tdom = wec.post_process(wec, results, waves, nsubsteps=nsubsteps)
        pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=nsubsteps)
        results = {
            'results': results,
            'pto_fdom': pto_fdom,
            'pto_tdom': pto_tdom,
            'wec_fdom': wec_fdom,
            'wec_tdom': wec_tdom,
            'pto': pto,
            'wec':wec,
            }
        return results
    
    #TODO
    def verification_rslts(self,flg_obj): 
        nfreq=self.nfreq
        waves_reg_fdom=self.waves_reg_fdom
        waves_irreg_fdom=self.waves_irreg_fdom
        g=self.g
        rho=self.rho
        waves_reg_fdom=self.waves_reg_fdom
        Zi=self.Zi
        hydro_data=self.hydro_data

        print("\nUnstructured, regular wave")
        verification_untructured_regular = self.verification(None, 1e-2, 2*nfreq+1, waves_reg_fdom,flg_obj)
        print("\nUnstructured, irregular wave")
        verification_untructured_irregular = self.verification(None, 1e-2, 2*nfreq+1, waves_irreg_fdom.sel(realization=[0]),flg_obj)
        print("\nPI, regular wave")
        verification_PI_regular = self.verification(wot.pto.controller_pi, 1e-2, 2, waves_reg_fdom,flg_obj)
        self.verification_untructured_regular=verification_untructured_regular
        self.verification_untructured_irregular=verification_untructured_irregular
        self.verification_PI_regular=verification_PI_regular

        ###############################
        # Regular Wave Calculations
        ###############################
        # Assume these are provided as attributes:
        # self.wavefreq_reg: wave frequency for regular wave [Hz]
        # self.amplitude_reg: wave amplitude for regular wave [m]
        wavefreq_reg = self.wavefreq_reg  
        amplitude_reg = self.amplitude_reg
        T_reg = 1 / wavefreq_reg                   # Period for regular wave [s]
        wavelength_reg = g * T_reg**2 / (2 * np.pi)  # Deep water wavelength [m]
        H_reg = 2 * amplitude_reg                  # Full wave height for regular wave [m]
        # For regular waves, the formula uses denominator 32*pi
        power_density_reg = (rho * g**2 / (32 * np.pi)) * (H_reg**2) * T_reg  # [W/m]
        max_power_available_reg = power_density_reg * (wavelength_reg / (2 * np.pi))  # [W]

        print("\nRegular Wave Metrics:")
        print(f"  Wavelength:           {wavelength_reg:.2f} m")
        print(f"  Wave power density:   {power_density_reg:.2f} W/m")
        print(f"  Max power available:  {max_power_available_reg:.2f} W")

        ###############################
        # Irregular Wave Calculations
        ###############################
        # Assume these are provided as attributes:
        # self.H_sig: significant wave height for irregular wave [m]
        # self.T_dom: dominant period for irregular wave [s]
        H_sig = self.H_sig       # Significant wave height for irregular waves
        T_dom = self.T_dom       # Dominant Energy period for irregular waves [s] #Te_irrreg
        Tp_irreg=self.Tp_irreg #Tp_irrreg
        wavelength_irreg = g * Tp_irreg**2 / (2 * np.pi)  # Deep water wavelength for irregular wave [m]
        # For irregular waves, the formula uses denominator 64*pi
        power_density_irreg = (rho * g**2 / (64 * np.pi)) * (H_sig**2) * T_dom  # [W/m]
        max_power_available_irreg = power_density_irreg * (wavelength_irreg / (2 * np.pi))  # [W]

        #max_power_available_irreg_2=(np.pi/4)*rho * g*H_sig/T_dom*self.V0_disp
        #(np.pi/4)*rho * g*H_reg/T_reg*self.V0_disp

        print("\nIrregular Wave Metrics:")
        print(f"  Wavelength:           {wavelength_irreg:.2f} m")
        print(f"  Wave power density:   {power_density_irreg:.2f} W/m")
        print(f"  Max power available:  {max_power_available_irreg:.2f} W")    
        
        #power_flow_reg=calculate_power_flows(verification_untructured_regular['wec'], verification_untructured_regular['pto'], verification_untructured_regular['results'], waves_reg_fdom, Zi)
        #plot_power_flow(power_flow_reg)
        #plt.savefig(f"{self.results_folder}/power_flow_reg.png", dpi=300, bbox_inches='tight')

        power_flow_irreg=calculate_power_flows(verification_untructured_irregular['wec'], verification_untructured_irregular['pto'], verification_untructured_irregular['results'], waves_irreg_fdom, Zi)
        plot_power_flow(power_flow_irreg)
        plt.savefig(f"{self.results_folder}/power_flow_irreg.png", dpi=300, bbox_inches='tight')

        max_abs_value_I = np.abs(verification_untructured_irregular['pto_tdom'][0]['trans_flo']).max()

        plt.show(block=True)

    ## Complex conjugate solution
    def thevenin_equivalent(self,intrinsic_impedance, pto_impedance, excitation_force):
        Z_11 = pto_impedance[0, 0, :]
        Z_12 = pto_impedance[0, 1, :]
        Z_21 = pto_impedance[1, 0, :]
        Z_22 = pto_impedance[1, 1, :]
        V_th =  Z_21 / (-Z_11 + intrinsic_impedance) * excitation_force  # Equation (19)  
        Z_th = Z_22 + (Z_12*Z_21) / (-Z_11 + intrinsic_impedance)  # Equation (20)
        return V_th, Z_th

    def complex_conjugate_solution(self,thev_voltage, thev_impedance):
        thev_current_opt = thev_voltage / (2*thev_impedance.real)
        cc_current = -1.0 *  thev_current_opt # Equation (21)    
        cc_voltage = thev_impedance.conj() * thev_current_opt  # Equation (22)
        return cc_current, cc_voltage

    def theory_rslts(self):
        # regular wave
        ## PTO impedance, WEC impedance, excitation forces
        verification_PI_regular=self.verification_PI_regular
        bem_data=self.bem_data
        nfreq=self.nfreq
        fb=self.fb
        omega=self.omega
        wec=self.wec
        Zi=self.Zi
        hydro_data=self.hydro_data
        waves_reg_fdom=self.waves_reg_fdom
        nsubsteps=self.nsubsteps
        results_folder=self.results_folder
        verification_untructured_irregular=self.verification_untructured_irregular
        verification_untructured_regular=self.verification_untructured_regular
        Fe_reg = (verification_PI_regular['wec_fdom'][0].sel(type = 'Froude_Krylov').force
                + verification_PI_regular['wec_fdom'][0].sel(type = 'diffraction').force
                ).sel(influenced_dof = 'DOF_0')
        #hydro_data=wot.add_linear_friction(bem_data=bem_data)
        #Zi = wot.hydrodynamic_impedance(hydro_data).sel(influenced_dof='Heave', radiating_dof='Heave')

        V_th_reg, Z_th =  self.thevenin_equivalent(Zi, self.pto_impedance(), Fe_reg)
        cc_current_fd_reg, cc_voltage_fd_reg = self.complex_conjugate_solution(V_th_reg, Z_th)

        # Equation (23) >>>
        cc_velocity_fd = np.zeros(nfreq) *0j
        cc_force_fd = np.zeros(nfreq) * 0j
        for ifreq in range(nfreq):
            abcd_inv = np.linalg.inv(wot.pto._make_abcd(self.pto_impedance(), fb.nb_dofs)[:, :, ifreq])
            vec_elec = np.array([[cc_current_fd_reg[ifreq]], [cc_voltage_fd_reg[ifreq]]])
            vec_mech = abcd_inv @ vec_elec
            cc_velocity_fd[ifreq] = vec_mech[0, 0]
            cc_force_fd[ifreq] = vec_mech[1, 0]
        cc_velocity_fd = xr.DataArray(cc_velocity_fd, dims=["omega"], coords=[("omega", omega),],)
        cc_force_fd = xr.DataArray(cc_force_fd, dims=["omega"], coords=[("omega", omega), ],)
        # Equation (23) <<<
        C_cc = cc_force_fd/cc_velocity_fd  # Equation (26)
        # Equation (27) >>>
        wave_frequency = waves_reg_fdom.attrs.get('Frequency (Hz)')
        wave_index = C_cc.omega.values.tolist().index(wave_frequency*2*np.pi)
        B_cc = np.real(C_cc.isel(omega = wave_index -0 ).values)
        # Equation (27) <<<
        K_cc = -1*wave_frequency*2*np.pi*np.imag(C_cc.isel(omega = wave_index - 0).values)  # Equation (28)

        # time-domain
        time = wec.time_nsubsteps(nsubsteps)
        time_xr = xr.DataArray(time)
        cc_current_td_reg = wot.time_results(cc_current_fd_reg, time_xr)
        cc_voltage_td_reg = wot.time_results(cc_voltage_fd_reg,  time_xr)
        cc_power_td_reg = cc_current_td_reg * cc_voltage_td_reg

        # Irregular wave
        Fe_irreg = (verification_untructured_irregular['wec_fdom'][0].sel(type = 'Froude_Krylov').force
                + verification_untructured_irregular['wec_fdom'][0].sel(type = 'diffraction').force
                ).sel(influenced_dof = 'DOF_0')
        
        V_th_irreg,_ =  self.thevenin_equivalent(Zi, self.pto_impedance(), Fe_irreg)
        cc_current_fd_irreg, cc_voltage_fd_irreg = self.complex_conjugate_solution(V_th_irreg, Z_th)

        # time-domain
        cc_current_td_irreg = wot.time_results(cc_current_fd_irreg, time_xr)
        cc_voltage_td_irreg = wot.time_results(cc_voltage_fd_irreg,  time_xr)
        cc_power_td_irreg = cc_current_td_irreg * cc_voltage_td_irreg

        # Gains
        B_wot, K_wot = verification_PI_regular['results'][0].x[-2:] #the last two are control opt avribales (after states)
        print(f'PI controller gains (theoretical): K = {K_cc:.1f} N/m, B = {B_cc:.1f} Ns/m')
        print(f'PI controller gains (WecOptTool): K = {K_wot:.1f} N/m, B  {B_wot:.1f} Ns/m')

        with open(f"{results_folder}/pi_controller_gains.txt", "w") as file:
            file.write(f'PI controller gains (theoretical): K = {K_cc:.1f} N/m, B = {B_cc:.1f} Ns/m\n')
            file.write(f'PI controller gains (WecOptTool):   K = {K_wot:.1f} N/m, B = {B_wot:.1f} Ns/m\n')

        self.Zi=Zi
        # Figure 4 (paper)
        t_plot = 10
        plt.figure()
        plt.plot(time, cc_power_td_reg, '-', color='0.5', linewidth=0.5, label='Theoretical optimal solution')
        plt.plot(time, verification_PI_regular['pto_tdom'][0].power[1], '+', label='PI controller')
        plt.plot(time, verification_untructured_regular['pto_tdom'][0].power[1], '--', label='Unstructured controller')
        plt.xlabel('Time [s]')
        plt.ylabel('Electrical power [W]')
        plt.legend(loc='upper center',)
        plt.ylim([-100, 100])
        plt.grid()
        plt.xlim([0, t_plot])
        plt.axhline(y=0, xmin = 0, xmax = 1, color = 'k', linewidth=0.5)
        plt.savefig(f'{results_folder}/IEEE_2023_verification_epower_reg.pdf')

        # Figure 5 (paper)
        plt.figure()
        plt.plot(time, cc_power_td_irreg, '-', color='0.5', linewidth=0.5, label='Theoretical optimal solution')
        plt.plot([],[])
        plt.plot(time, verification_untructured_irregular['pto_tdom'][0].power[1], '--', label='Unstructured controller')
        plt.xlabel('Time [s]')
        plt.ylabel('Electrical power [W]')
        plt.legend(loc='upper center',)
        plt.ylim([-4000, 4000])
        plt.grid()
        plt.xlim([0, t_plot*2])
        plt.axhline(y=0, xmin = 0, xmax = 1, color = 'k', linewidth=0.5)
        plt.savefig(f'{results_folder}/IEEE_2023_verification_epower_irreg.pdf')

    def outer_opt(self,x,f_max,scale_x_opt,sea_state_power,N):
        fb=self.fb
        nsubsteps=self.nsubsteps
        wec=self.wec
        waves=self.waves
        nfreq=self.nfreq
        sea_states_labels=self.sea_states_labels
        Zi=self.Zi
        #unpack optimization variables
        drivetrain_inertia = x[0]
        drivetrain_stiffness = x[1]
        global count
        count +=1
        # PTO
        pto = wot.pto.PTO(
            fb.nb_dofs,
            np.eye(fb.nb_dofs),
            None,
            self.pto_impedance(drivetrain_inertia, drivetrain_stiffness), #None
            None,
            ["PTO_Heave"],
        )
        # PTO force constraint constraints
        def const_f_pto(wec, x_wec, x_opt, waves):
            f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
            return f_max - np.abs(f.flatten())
        """wec.constraints = [{'type': 'ineq',
                            'fun': const_f_pto,
                            }]"""
        # WEC additional forces
        wec.forces['PTO'] = pto.force_on_wec
        # objective function
        objective = pto.average_power  #pto.mechanical_average_power    # pto.average_power

        # run each sea state
        print(f"\nPTO {count}/{N}. m: {drivetrain_inertia:.1f}, k:{drivetrain_stiffness:.1f}",
            f" sc_x_opt:{scale_x_opt}")
        average_annual_power = 0
        for iw, wave in enumerate(waves):
            results = wec.solve(wave, objective, 2*nfreq+1, scale_x_opt=scale_x_opt, optim_options={"disp": False,"maxiter":200})
            avg_power = results[0].fun
            sea_state_power[count-1][iw] = avg_power
            average_annual_power = average_annual_power + avg_power * wave.weight
            print(f"  sea state: {iw}-{sea_states_labels[iw]}, " +
                f"exit mode: {results[0].status}, nit: {results[0].nit}, avg. power: {avg_power:.2f}W")
            
            # post-process
            """wec_fdom, wec_tdom = wec.post_process(wec, results, wave, nsubsteps=nsubsteps)
            pto_fdom, pto_tdom = pto.post_process(wec, results, wave, nsubsteps=nsubsteps)
            results = {
                'results': results,
                'pto_fdom': pto_fdom,
                'pto_tdom': pto_tdom,
                'wec_fdom': wec_fdom,
                'wec_tdom': wec_tdom,
                'pto': pto,
                'wec':wec,
                }
            power_flow_reg=calculate_power_flows(results['wec'], results['pto'], results['results'], wave, Zi)
            print("power_flow_reg['Electrical (solver)']\n", power_flow_reg['Electrical (solver)'])
            print("power_flow_reg['Mechanical (solver)']\n", power_flow_reg['Mechanical (solver)'])
            plot_power_flow(power_flow_reg)"""
            


        print(f"Average annual power: {average_annual_power:.2f}W")
        return average_annual_power
    
    def Run_outer_opt(self,f_max=8000):
        results_folder=self.results_folder
        global count
        # brute optimization parameter space
        waves=self.waves
        drivetrain_stiffness_list =np.linspace(-15, 15, 7)# np.linspace(-15, 15, 7)  # default: 0.0
        drivetrain_inertia_list =np.linspace(0,26, 14)# np.linspace(0, 26, 14)  # default: 2.0
        def list_to_range(l1):
            if len(l1) >1:
                return slice(l1[0], l1[-1]+np.diff(l1)[0], np.diff(l1)[0])
            else:
                return slice(l1[0], 1.5*l1[0], l1[0])
        ranges = (list_to_range(drivetrain_inertia_list),
                list_to_range(drivetrain_stiffness_list),)

        scale_x_opt=1e-2
        N = len(drivetrain_inertia_list) * len(drivetrain_stiffness_list)
        sea_state_power = [[0]*len(waves) for i in range(N)]
        count = 0
        res = brute(func=self.outer_opt,
                    args=(f_max,scale_x_opt,sea_state_power,N),
                    ranges=ranges,
                    full_output=True,
                    finish=None)
        with open(f'{results_folder}/result_opt_brute.pkl', 'wb') as f:
            pkl.dump(res, f)
            pkl.dump(sea_state_power,f)

    def plot_Run_outer_opt(self):
        results_folder=self.results_folder
        drivetrain_stiffness_list =np.linspace(-15, 15, 7)
        drivetrain_inertia_list =np.linspace(0, 26, 14)
        sea_states=self.sea_states
        waves=self.waves
        wec=self.wec
        nsubsteps=self.nsubsteps
        fb=self.fb
        nfreq=self.nfreq
        Zi=self.Zi
        spectra=self.spectra
        f1=self.f1
        freq=self.freq
        sea_states_labels=self.sea_states_labels
        with open(f'{results_folder}/result_opt_brute.pkl', 'rb') as f:
            res = pkl.load(f)
            sea_state_power=pkl.load(f)
        # Plot
        fig, ax = plt.subplots(ncols=1, figsize=(7, 6))
        (x, y, z)  = (res[2][1], res[2][0], -1*res[3])
        pcm = ax.pcolormesh(x, y, z, cmap=cm.magma, edgecolor='w' )
        ax.set_ylabel('Drivetrain inertia [kg m$^2$]', fontsize = 14)
        ax.set_yticks(drivetrain_inertia_list)
        ax.set_xlabel('Drivetrain stiffness [N m/rad]', fontsize = 14)
        ax.set_xticks(drivetrain_stiffness_list)
        pcb = fig.colorbar(pcm, ax=ax)
        pcb.set_label('Average electrical power [W]', fontsize = 14)
        wx = x[0,1]-x[0,0]
        wy = y[1,0] - y[0,0]
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i, j] == -1*res[1]: #optimal
                    n_opt = i*z.shape[1] + j
                    color = 'C1'
                    ax.add_patch(plt.Rectangle((x[i,j]-wx/2, y[i,j]-wy/2), wx, wy, fc='none', ec=color, lw=3, clip_on=False))
                elif res[2][0][i,j] == 2 and res[2][1][i,j] == 0: #nominal configuration
                    n_nominal = i*z.shape[1] + j
                    P_nominal = z[i,j]
                    color = 'C2'
                    ax.add_patch(plt.Rectangle((x[i,j]-wx/2, y[i,j]-wy/2), wx, wy, fc='none', ec=color, lw=3, clip_on=False, linestyle = '--'))
        plt.savefig(os.path.join(results_folder, "Beaver_Island_power_contour_PS.pdf"), format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Optimal power {-res[1]:.1f}',
            f'\nNominal power {P_nominal:.1f}',
            f'\nImprovement: {((-res[1]/P_nominal - 1) *100):.1f}%')

        sea_state_power_optimal = np.round(sea_state_power[n_opt],0)
        sea_state_power_nominal = np.round(sea_state_power[n_nominal],0)
        sea_state_improvment = (np.round(sea_state_power_optimal/sea_state_power_nominal,3) - 1)*100

        print(f'Optimal configuration sea state power {sea_state_power_optimal}'   )
        print(f'Nominal configuration sea state power {sea_state_power_nominal}'   )
        print(f'Optimal config % improvement  {sea_state_improvment}'   )

        #fig.savefig('IEEE_2023_power_contour_PS.pdf', bbox_inches='tight')
        

        # Table II
        # save in latex format
        filename = "table_sea_state_power"
        tmp = '\t'
        with open(filename, "w") as f:
            for i, sea_state in sea_states.iterrows():
                f.write(" "*8 + f"{sea_states_labels[i]}  & {sea_state.Hm0:.2f} & {sea_state.Te:.2f} & {sea_state.weight*100:.1f} & {-sea_state_power_optimal[i]:.0f} & {-sea_state_power_nominal[i]:.0f} & {(sea_state_improvment[i] - 1)*100:.1f}  \\\\\n")
            f.write(" "*8 + f"\\bottomrule \n " + " "*8 + f"\\multicolumn{{4}}{{c||}}{{\\textbf{{Annual}}}} & \\textbf{{{-res[1]:.0f}}} & \\textbf{{{P_nominal:.0f}}} & \\textbf{{{((-res[1]/P_nominal - 1) *100):.1f}}} \\\\\n")
        # print
        print(" # | Hm0 [m] | Te [s] | w [%] || Popt [W] | Pnom [W] | imp. [%]")
        for i, sea_state in sea_states.iterrows():
            print(f"  {sea_states_labels[i]},  {sea_state.Hm0:.2f},  {sea_state.Te:.2f},  {sea_state.weight*100:.1f},  {-sea_state_power_optimal[i]:.0f},  {-sea_state_power_nominal[i]:.0f},  {(sea_state_improvment[i] - 1)*100:.1f}\n")
        print(f"\nAnnual {-res[1]:.0f}, - , - , - , {P_nominal:.0f},  {((-res[1]/P_nominal - 1) *100):.1f}\n")

        with open(f"{results_folder}/output_results.txt", "w") as f:
            f.write(f'Optimal power {-res[1]:.1f}\n')
            f.write(f'Nominal power {P_nominal:.1f}\n')
            f.write(f'Improvement: {((-res[1]/P_nominal - 1) *100):.1f}%\n\n')

            sea_state_power_optimal = np.round(sea_state_power[n_opt], 0)
            sea_state_power_nominal = np.round(sea_state_power[n_nominal], 0)
            sea_state_improvment = (np.round(sea_state_power_optimal/sea_state_power_nominal, 3) - 1)*100

            f.write(f'Optimal configuration sea state power: {sea_state_power_optimal}\n')
            f.write(f'Nominal configuration sea state power: {sea_state_power_nominal}\n')
            f.write(f'Optimal config % improvement: {sea_state_improvment}\n\n')

            f.write(" # | Hm0 [m] | Te [s] | w [%] || Popt [W] | Pnom [W] | imp. [%]\n")
            for i, sea_state in sea_states.iterrows():
                f.write(f"  {sea_states_labels[i]},  {sea_state.Hm0:.2f},  {sea_state.Te:.2f},  {sea_state.weight*100:.1f},  {-sea_state_power_optimal[i]:.0f},  {-sea_state_power_nominal[i]:.0f},  {(sea_state_improvment[i] - 1)*100:.1f}\n")
            f.write(f"\nAnnual {-res[1]:.0f}, - , - , - , {P_nominal:.0f},  {((-res[1]/P_nominal - 1) *100):.1f}\n")

        """### IV.B.1: Time Series Results"""

        # solve nominal and optimal cases
        def var_pto_solve(pto_impedance, scale_x_opt, waves):
            # PTO
            pto = wot.pto.PTO(
                fb.nb_dofs,
                np.eye(fb.nb_dofs),
                None,
                pto_impedance,
                None,
                ["PTO_Heave"],
            )
            # WEC additional forces
            wec.forces['PTO'] = pto.force_on_wec
            # objective function
            objective = pto.average_power
            # optimal controller
            results = wec.solve(waves, objective, 2*nfreq+1, scale_x_opt=scale_x_opt)
            # post-process
            pto_fdom, pto_tdom = pto.post_process(wec, results, waves, nsubsteps=nsubsteps)
            wec_fdom, wec_tdom = wec.post_process(wec,results, waves, nsubsteps=nsubsteps)
            results = {
                'results': results,
                'pto_fdom': pto_fdom,
                'pto_tdom': pto_tdom,
                'wec_fdom': wec_fdom,
                'wec_tdom': wec_tdom}
            return results

        opt_pto_config = res[0].tolist()
        pto_impedance_optimal = self.pto_impedance(*opt_pto_config)
        pto_impedance_nominal = self.pto_impedance()
        optimal_pto_irreg = var_pto_solve(pto_impedance_optimal, 1e-2,  waves[0])
        nominal_pto_irreg = var_pto_solve(pto_impedance_nominal, 1e-2,  waves[0])

        # post-process: current and voltage
        def get_current_voltage(pto_impedance, res):
            pto = wot.pto.PTO(fb.nb_dofs, np.eye(fb.nb_dofs), None, pto_impedance, None, ["PTO_Heave"])
            x_wec, x_opt = wot.decompose_state(res.x, fb.nb_dofs, nfreq)
            q1_td = pto.velocity(wec, x_wec, x_opt, waves[0])
            e1_td = pto.force(wec, x_wec, x_opt, waves[0])
            q1 = wot.complex_to_real(wot.td_to_fd(q1_td, False))
            e1 = wot.complex_to_real(wot.td_to_fd(e1_td, False))
            vars_1 = np.hstack([q1, e1])
            vars_1_flat = wot.dofmat_to_vec(vars_1)
            vars_2_flat = np.dot(pto.transfer_mat, vars_1_flat)
            vars_2 = wot.vec_to_dofmat(vars_2_flat, 2*pto.ndof)
            q2 = vars_2[:, :pto.ndof]
            e2 = vars_2[:, pto.ndof:]
            time_mat = pto._tmat(wec, nsubsteps)
            q2_td = np.dot(time_mat, q2)
            e2_td = np.dot(time_mat, e2)
            return e2_td, q2_td, wot.real_to_complex(e2, True), wot.real_to_complex(q2, True)
        voltage_nom_td, current_nom_td, voltage_nom_fd, current_nom_fd = get_current_voltage(pto_impedance_nominal, nominal_pto_irreg['results'][0])
        voltage_opt_td, current_opt_td, voltage_opt_fd, current_opt_fd = get_current_voltage(pto_impedance_optimal, optimal_pto_irreg['results'][0])

        # theoretical results
        time = wec.time_nsubsteps(nsubsteps)
        time_xr = xr.DataArray(time)

        Fe_nom_irr_A = (nominal_pto_irreg['wec_fdom'][0].force.sel(type = 'Froude_Krylov') +
                nominal_pto_irreg['wec_fdom'][0].force.sel(type = 'diffraction')).sel(influenced_dof = 'DOF_0')
        Fe_opt_irr_A = (optimal_pto_irreg['wec_fdom'][0].force.sel(type = 'Froude_Krylov') +
                optimal_pto_irreg['wec_fdom'][0].force.sel(type = 'diffraction')).sel(influenced_dof = 'DOF_0')

        V_th_opt, Z_th_opt =  self.thevenin_equivalent(Zi,pto_impedance_optimal,Fe_opt_irr_A)
        V_th_nom, Z_th_nom =  self.thevenin_equivalent(Zi,pto_impedance_nominal,Fe_nom_irr_A)

        v_th_nom_td = wot.time_results(V_th_nom, time_xr)
        v_th_opt_td = wot.time_results(V_th_opt, time_xr)
        fe_nom_irr_A_td = wot.time_results(Fe_nom_irr_A, time_xr)
        fe_opt_irr_A_td = wot.time_results(Fe_opt_irr_A, time_xr)

        power_max = np.abs(Fe_nom_irr_A)**2 / (8* np.real(Zi))
        epower_nom_max = np.abs(V_th_nom)**2 / (8* np.real(Z_th_nom))
        epower_opt_max = np.abs(V_th_opt)**2 / (8* np.real(Z_th_opt))

        # Plot
        def align_yyaxis(ax1, ax2):
            ax1_ylims = ax1.axes.get_ylim()
            ax1_yratio = ax1_ylims[0] / ax1_ylims[1]
            ax2_ylims = ax2.axes.get_ylim()
            ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
            if ax1_yratio < ax2_yratio:
                ax2.set_ylim(bottom = ax2_ylims[1]*ax1_yratio)
            else:
                ax1.set_ylim(bottom = ax1_ylims[1]*ax2_yratio)

        # plot
        fig, ax = plt.subplots(nrows=4,
                            figsize=(6,12), sharex = True,
                            constrained_layout=True)

        # subplot 1
        line1 = ax[0].plot(time, fe_nom_irr_A_td/1000, ':', label='Excitation force')
        ax[0].set_ylabel('Force [kN]')
        ax0r = ax[0].twinx()
        line2 = ax0r.plot(time, optimal_pto_irreg['wec_tdom'][0].vel.sel(influenced_dof = 'DOF_0'), 'C1', label = 'WEC velocity (Optimal)')
        line3 = ax0r.plot(time, nominal_pto_irreg['wec_tdom'][0].vel.sel(influenced_dof = 'DOF_0'), 'C2', label = 'WEC velocity (Nominal)', linestyle='--')
        ax0r.set_ylabel('Velocity [ms] ')
        ax0r.tick_params(axis='y', color='black', labelcolor='black')
        align_yyaxis(ax[0],ax0r)
        lines = line1 + line2 + line3
        ax0r.legend(lines, ['Excitation force','WEC velocity (Optimal)', 'WEC velocity (Nominal)' ])
        plt.axhline(y=0, xmin = 0, xmax = 1, color = '0.75', linewidth=0.5)
        ax[0].grid(color='0.75', linestyle='-',
                            linewidth=0.5, axis = 'x')

        # subplot 2
        line1 = ax[1].plot(time, v_th_opt_td, 'C0', linestyle = 'dotted')
        ax[1].set_ylabel('Voltage [V]')
        ax1r = ax[1].twinx()
        line2 = ax1r.plot(time, -1*current_opt_td, 'C1')
        ax1r.set_ylabel('Current [A] ')
        ax1r.tick_params(axis='y', color='black', labelcolor='black')
        align_yyaxis(ax[1],ax1r)
        lines = line1  + line2
        ax[1].legend(lines, ['Thevenin voltage - excitation (Optimal)',
                            '-1 * PTO current (Optimal)'])
        plt.axhline(y=0, xmin = 0, xmax = 1, color = '0.75', linewidth=0.5)
        ax[1].grid(color='0.75', linestyle='-', linewidth=0.5, axis = 'x')
        ax[1].set_ylim([-250, 250])
        ax1r.set_ylim([-80, 80])

        # subplot 3
        line1 = ax[2].plot(time, v_th_nom_td, 'C0', linestyle = 'dotted')
        ax[2].set_ylabel('Voltage [V]')
        ax2r = ax[2].twinx()
        line2 = ax2r.plot(time, -1*current_nom_td, 'C2', linestyle='--')
        ax2r.set_ylabel('Current [A] ')
        ax2r.tick_params(axis='y', color='black', labelcolor='black')
        align_yyaxis(ax[2],ax2r)
        lines =  line1 +line2
        ax[2].legend(lines, ['Thevenin voltage - excitation (Nominal)',
                            '-1 * PTO current (Nominal)'])
        plt.axhline(y=0, xmin = 0, xmax = 1, color = '0.75', linewidth=0.5)
        ax[2].grid(color='0.75', linestyle='-', linewidth=0.5, axis = 'x')
        ax[2].set_ylim([-250, 250])
        ax2r.set_ylim([-80, 80])

        # subplot 4
        line1 = ax[3].plot(time, optimal_pto_irreg['pto_tdom'][0].force/1000, 'C1', label = 'PTO force (Optimal)')
        line2 = ax[3].plot(time, nominal_pto_irreg['pto_tdom'][0].force/1000, 'C2', label = 'PTO force (Nominal)', linestyle='--')
        ax[3].set_ylabel('Force [kN] ')
        ax[3].axhline(y=8, linestyle=':', linewidth=1.5, color = 'k')
        ax[3].legend(['PTO force (Optimal)', 'PTO force (Nominal)', 'PTO force limit'])
        ax[3].axhline(y=-8, linestyle=':', linewidth=1.5, color = 'k')
        ax[3].axhline(y=0, xmin = 0, xmax = 1, color = '0.75', linewidth=0.5)
        ax[3].grid(color='0.75', linestyle='-', linewidth=0.5, axis = 'x')
        ax[3].set_xlabel('Time [s]')
        ax[3].set_xlim([0, 60])

        #fig.savefig('IEEE_2023_timeseries_SeaState_A_effort_flow.pdf', bbox_inches='tight')
        plt.savefig(os.path.join(results_folder, "timeseries_SeaState_A_effort_flow.pdf"), format='pdf', dpi=300, bbox_inches='tight')

        """### IV.B.2: Frequency Domain Results"""

        fig, ax = plt.subplots(nrows = 2, figsize=(6,5), constrained_layout=True, sharex= True)

        # subplot 1
        ax[0].plot(freq,spectra[0]/spectra[0].max(), linestyle = ':', label= f'JONSWAP spectum sea state {sea_states_labels[0]}')
        Fe_irr_A = Fe_opt_irr_A  # Fe_nom_irr_A
        ax[0].plot(freq,np.abs(Fe_irr_A[1:])/np.abs(Fe_irr_A).max(),  color ='grey', linestyle = '-.', label = f'Excitation force spectrum sea state {sea_states_labels[0]}' )
        ax[0].plot(freq, np.abs(V_th_opt)/np.abs((V_th_opt)).max().squeeze(), label = 'Thevenin voltage (Optimal)', linestyle = '-')
        ax[0].plot(freq, np.abs(V_th_nom)/np.abs((V_th_nom)).max().squeeze(), label = 'Thevenin voltage (Nominal)', linestyle = '--')
        ax[0].set_ylabel('')
        ax[0].set_ylim(bottom=0)
        ax[0].set_title('')
        ax[0].set_xlim(left=f1, right=f1*nfreq)
        ax[0].legend(loc = 'upper right')

        # subplot 2
        markerline_opt, _, _, = ax[1].stem(freq,-0.5*np.real(np.conj(current_opt_fd[1:])*voltage_opt_fd[1:]), linefmt = "C1", markerfmt= 'C1o', label = 'Average electrical power (Optimal)', basefmt=" ")
        ax[1].plot(freq,epower_opt_max.squeeze(),color = 'k',linestyle = '-', label = 'Maximum electrical power (Optimal)')
        plt.setp(markerline_opt, markersize = 4)
        markerline_nom, _, _, = ax[1].stem(freq,-0.5*np.real(np.conj(current_nom_fd[1:])*voltage_nom_fd[1:]), linefmt = "C2", markerfmt= 'C2x', label = 'Average electrical power (Nominal)', basefmt=" ")
        plt.setp(markerline_nom, markersize = 6)
        ax[1].plot(freq,epower_nom_max.squeeze(), 'k--', label = 'Maximum electrical power (Nominal)')

        # labels
        ax[0].set_xlabel('')
        ax[1].set_xlabel('Frequency [Hz]')
        ax[1].set_ylabel('Electrical power [W]')
        ax[1].legend()
        fig.savefig(os.path.join(results_folder,'input_spectra.pdf'), format='pdf', dpi=300, bbox_inches='tight')

        # figure 2
        fig, ax = plt.subplots(nrows=1, figsize=(6,2), sharex = True, constrained_layout=True)
        ax.stem(freq, (np.angle(V_th_opt/(-1*current_opt_fd[1:, 0])))*180/np.pi, linefmt = "C1",
                markerfmt= 'C1o', label = "Thevenin voltage / (-1 * PTO current (Optimal))", basefmt ='none')
        ax.stem(freq, (np.angle(V_th_nom/(-1*current_nom_fd[1:, 0])))*180/np.pi, linefmt = "C2--", markerfmt= 'C2x', label = "Thevenin voltage / (-1 * PTO current (Nominal))", basefmt ='none')
        ax.legend()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Phase [degree]')
        ax.set_xlim([0.08, 0.42])
        ax.set_ylim([-180, 180])

        #fig.savefig('IEEE_2023_phase_matching.pdf', bbox_inches='tight')
        plt.savefig(os.path.join(results_folder, "phase_matching.pdf"), format='pdf', dpi=300, bbox_inches='tight')






if __name__ == '__main__':
    saved_rslts_addrs="Rslts_1"   #  "Rslts_3(253) "   #   "Rslts_2(243) "#"  Rslts_1(269) "
    optimizer = wecOpt(saved_rslts_addrs)
    station_number ="269"      #253 #"243"# "269"
    optimizer.load_Wave_Data(station_number)
    optimizer.plot_Wave_Data()
    optimizer.sea_state_clustering(N=10) #N: sea_state_clustering numbers
    optimizer.plot_Sea_States()
    optimizer.compare_Meshes()
    optimizer.Load_WEC()
    amplitude_reg = 0.94 #0.0625
    wavefreq_reg = 1/8.75 #0.3
    optimizer.reg_wave(amplitude_reg,wavefreq_reg)
    Te_irreq = 7.905 #11.162585# 7.905384 #7.62
    hs_irreq = 0.9415 #3.454921# 0.941541 #1.5
    optimizer.irreg_wave(Te_irreq,hs_irreq)
    optimizer.verification_rslts(flg_obj="mech")
    optimizer.theory_rslts()
    optimizer.Run_outer_opt(f_max=8000)
    optimizer.plot_Run_outer_opt()
