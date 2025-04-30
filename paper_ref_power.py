import numpy as np
import pandas as pd
import mhkit.wave as wave  # Make sure MHKiT is installed

# Physical constants
rho = 1025.0   # kg/m^3
g = 9.81       # m/s^2
gamma = 3.3    # JONSWAP gamma (typical value)

# Sea state data from the uploaded image
sea_states = pd.DataFrame({
    'label':  ['A','B','C','D','E','F','G','H','I','J'],
    'Hm0':    [1.48, 1.58, 1.58, 1.91, 2.73, 2.94, 2.96, 3.70, 4.79, 5.96],
    'Te':     [7.63, 9.15, 6.31,11.11, 8.10, 9.77,13.93,11.53, 9.69,12.54],
    'weight': [19.6,14.9,15.8, 8.4, 11.5,11.6, 3.4, 7.2, 5.1, 2.7],
})

# 1. Compute power density [kW/m]
sea_states["power_density"] = (
    (rho * g**2) / (64 * np.pi) *
    sea_states["Hm0"]**2 *
    sea_states["Te"] / 1000  # Convert from W/m to kW/m
)

# 2. Compute Tp from Te using MHKiT
sea_states["Tp"] = wave.resource.energy_period_to_peak_period(sea_states["Te"], gamma)

# 3. Compute wavelength and capture width
sea_states["lambda"] = g * sea_states["Tp"]**2 / (2 * np.pi)
sea_states["capture_width"] = sea_states["lambda"] / (2 * np.pi)

# 4. Compute available power
sea_states["weight_frac"] = sea_states["weight"] / 100
sea_states["avail_power"] = (
    sea_states["power_density"] *
    sea_states["capture_width"] *
    sea_states["weight_frac"]
)

# 5. Compute requested sums
sum_power_density_weight = (sea_states["power_density"] * sea_states["weight_frac"]).sum()
sum_avail_power = sea_states["avail_power"].sum()

# 6. Print detailed table
pd.set_option("display.precision", 3)
print(sea_states[[
    "label", "Hm0", "Te", "Tp", "weight", 
    "power_density", "lambda", "capture_width", "avail_power"
]])

# 7. Print the summary totals
print("\n--- Summary Totals ---")
print(f"Sum of power_density * weight_frac = {sum_power_density_weight:.3f} [kW·%]")
print(f"Sum of available power       = {sum_avail_power:.3f} [kW]")
