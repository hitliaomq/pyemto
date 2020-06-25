import numpy as np
from pyemto.EOS import EOS
from pyemto.emto_parser import EMTOPARSER

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lat = 'bcc'
Natom = 2

ry2J = 2.1798741e-18
A2m = 1e-10
bohr2A = 0.529177249


# Equilibrium lattice constant output files
vol_data = EMTOPARSER('./kgrn/*', './kfcd/*', suffix='prn', DLM=False)

vol_data.create_df()
vol_df = vol_data.main_df

eos = EOS('test', method='morse')

# Bmod = 187 (previous EMTO)
#indMin = 3
#indMax = -7#len(vol_df.SWS)

# Bmod = 160 (previous VASP)
# The index used in the fitting
indMin = 0
indMax = len(vol_df.SWS)

SWS0, E0, B0, grun0, error0 = eos.fit(vol_df.SWS[indMin:indMax], vol_df.EPBE[indMin:indMax], show_plot=False)
vol0 = 4./3*np.pi*(SWS0*bohr2A)**3
a = (vol0 * Natom) ** (1./3.)

print('SWS0 = ', SWS0)
print('B    = ', B0)
print('a    = ', a)
