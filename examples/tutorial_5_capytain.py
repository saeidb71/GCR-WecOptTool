import capytaine as cpt
import numpy as np
#cpt.set_logging('DEBUG')#'INFO')
#from capytaine.post_pro import impedance
import xarray as xr
from capytaine import BEMSolver
from capytaine.bodies.predefined.spheres import Sphere
from capytaine.post_pro import impedance
from capytaine.post_pro import rao

#sphere = cpt.mesh_sphere(radius=1.0, center=(0, 0, -0.3), name="my sphere")
#sphere.show()

rigid_sphere = cpt.FloatingBody(
        mesh=cpt.mesh_sphere(radius=1.0, center=(0.0, 0.0, -0.0)),
        dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, -0.0)),
        center_of_mass=(0, 0, -0.0)
        ).immersed_part()

anim = rigid_sphere.animate(motion={"Heave": 0.1}, loop_duration=1.0)
anim.run()

# can use mesh.show()
print("Volume:", rigid_sphere.volume) #displaced volume == hydrostatics["disp_volume"]
print("Center of buoyancy:", rigid_sphere.center_of_buoyancy)
print("Wet surface area:", rigid_sphere.wet_surface_area)
print("Displaced mass:", rigid_sphere.disp_mass(rho=1025))
print("Waterplane center:", rigid_sphere.waterplane_center)
print("Waterplane area:", rigid_sphere.waterplane_area)
print("Metacentric parameters:",
    rigid_sphere.transversal_metacentric_radius,
    rigid_sphere.longitudinal_metacentric_radius,
    rigid_sphere.transversal_metacentric_height,
    rigid_sphere.longitudinal_metacentric_height)

hydrostatics = rigid_sphere.compute_hydrostatics()
print(hydrostatics.keys())

hydrostatics = rigid_sphere.compute_hydrostatics(rho=1025.0)

print(hydrostatics["disp_volume"])

# compute intinsic impedance

f = np.linspace(0.1, 2.0)
omega = 2*np.pi*f
rho_water = 1e3
r = 1

solver = BEMSolver()
test_matrix = xr.Dataset(coords={
    'rho': rho_water,
    'water_depth': [np.inf],
    'omega': omega,
    'wave_direction': 0,
    'radiating_dof': list(rigid_sphere.dofs.keys()),
    })

data = solver.fill_dataset(test_matrix, rigid_sphere,
                           hydrostatics=True,
                           mesh=True,
                           wavelength=True,
                           wavenumber=True)
Zi = impedance(data)
print(Zi)

rao = rao(data)
k=1
# 3.82267415555807

#print(hydrostatics["hydrostatic_stiffness"])
# <xarray.DataArray 'hydrostatic_stiffness' (influenced_dof: 6, radiating_dof: 6)> Size: 288B
# [...]
# Coordinates:
#   * influenced_dof  (influenced_dof) <U5 120B 'Surge' 'Sway' ... 'Pitch' 'Yaw'
#   * radiating_dof   (radiating_dof) <U5 120B 'Surge' 'Sway' ... 'Pitch' 'Yaw'

#print(hydrostatics["inertia_matrix"])
# <xarray.DataArray 'inertia_matrix' (influenced_dof: 6, radiating_dof: 6)> Size: 288B
# [...]
# Coordinates:
#   * influenced_dof  (influenced_dof) <U5 120B 'Surge' 'Sway' ... 'Pitch' 'Yaw'
#   * radiating_dof   (radiating_dof) <U5 120B 'Surge' 'Sway' ... 'Pitch' 'Yaw'
