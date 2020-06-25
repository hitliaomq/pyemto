import numpy as np
from pyemto.EOS import EOS
from pyemto.emto_parser import EMTOPARSER

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lat = 'bcc'

ry2J = 2.1798741e-18
A2m = 1e-10
bohr2A = 0.529177249

deltas = np.linspace(0, 0.05, 6)
deltas[0] = 0.001

# Equilibrium lattice constant output files
vol_data = EMTOPARSER('eqv/kgrn/*', 'eqv/kfcd/*', suffix='prn', DLM=False)

vol_data.create_df()
vol_df = vol_data.main_df

# Elastic constants output files
ec_data = EMTOPARSER('elastic/kgrn/*', 'elastic/kfcd/*', suffix='prn', DLM=False)

ec_data.create_df()
ec_df = ec_data.main_df


eos = EOS('test', method='morse')

# Bmod = 187 (previous EMTO)
#indMin = 3
#indMax = -7#len(vol_df.SWS)

# Bmod = 160 (previous VASP)
indMin = 0
indMax = len(vol_df.SWS)

SWS0, E0, B0, grun0, error0 = eos.fit(vol_df.SWS[indMin:indMax], vol_df.EPBE[indMin:indMax], show_plot=False)
vol0 = 4./3*np.pi*(SWS0*bohr2A)**3

plt.figure(figsize=(12, 6))
# d1 is the c' distortion
d1 = np.asarray(ec_df[ec_df.Struc.str.contains('d1')].EPBE)
plt.plot(deltas, d1, 'o-')

# d2 is the c44 distortion
d2 = np.asarray(ec_df[ec_df.Struc.str.contains('d2')].EPBE)
plt.plot(deltas, d2, 'o-')

plt.xlabel('$\delta$', fontsize=22)
plt.title('Energy vs. delta', fontsize=22)

plt.savefig('elastic_fitting.png')

cprime_coeffs, cprime_error = eos.distortion_fit(deltas, d1, num=1)
c44_coeffs, c44_error = eos.distortion_fit(deltas, d2, num=1)

# Change units to GPa
cprime_coeffs = cprime_coeffs[0] * ry2J / (vol0*A2m**3) / 2 / 1e9
c44_coeffs = c44_coeffs[0] * ry2J / (vol0*A2m**3) / 2 / 1e9

print('SWS0 = {sws0}bohr'.format(sws0=str(SWS0)))
print('B    = {B}'.format(B=str(B0)))
print('C\'  = {Cpre}'.format(Cpre=str(cprime_coeffs)))
print('C44  = {C44}'.format(C44=str(c44_coeffs)))
print('CP   = {CP}'.format(CP=str(B0 - 2./3*cprime_coeffs - c44_coeffs)))