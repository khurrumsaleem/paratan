e_neutron_dd = 2.45e6 #eV
e_neutron_dt = 14.1e6 #eV

e_neutron_dd_MJ = e_neutron_dd * 1.602176634e-19 / 1e6
e_neutron_dt_MJ = e_neutron_dt * 1.602176634e-19 / 1e6

length_end_plug_dd = 3.6 #m

power_density_dd = 0.1 #MW/m

total_power_dd_plugs = power_density_dd * length_end_plug_dd * 2 #MW

neutron_source_strength_dd = total_power_dd_plugs / e_neutron_dd_MJ

print(f"Neutron source strength for DD plugs: {neutron_source_strength_dd} neutrons/s")

dd_end_plug_neutron_fraction = 0.037

total_neutrons_system = neutron_source_strength_dd / dd_end_plug_neutron_fraction

print(f"Total neutrons in system: {total_neutrons_system} neutrons/s")

tandem_power_fraction = 0.9630

tandem_neutron_source_strength = total_neutrons_system * tandem_power_fraction

print(f"Tandem neutron source strength: {tandem_neutron_source_strength} neutrons/s")

tandem_neutron_power = tandem_neutron_source_strength * e_neutron_dt_MJ

print(f"Tandem neutron power: {tandem_neutron_power} MW")

total_power_system = tandem_neutron_power / 0.8
print(f"Tandem total power: {total_power_system} MW")

