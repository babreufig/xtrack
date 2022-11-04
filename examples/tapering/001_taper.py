import json

import numpy as np
from scipy.constants import c as clight
import xtrack as xt

with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

line_df = line.to_pandas()
multipoles = line_df[line_df['element_type'] == 'Multipole']
cavities = line_df[line_df['element_type'] == 'Cavity'].copy()

# save voltages
cavities['voltage'] = [cc.voltage for cc in cavities.element.values]
cavities['frequency'] = [cc.frequency for cc in cavities.element.values]
cavities['eneloss_partitioning'] = cavities['voltage'] / cavities['voltage'].sum()


# set voltages to zero
for cc in cavities.element.values:
    cc.voltage = 0

tracker = xt.Tracker(line = line)
tw_no_rad = tracker.twiss(method='4d')

p_test = tw_no_rad.particle_on_co.copy()
tracker.configure_radiation(mode='mean')

tracker.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
mon = tracker.record_last_track

n_cavities = len(cavities)

tracker_taper = xt.Tracker(line = line, extra_headers=["#define XTRACK_MULTIPOLE_TAPER"])

import matplotlib.pyplot as plt

rtot_eneloss = 1e-10

# Put all cavities on crest and at zero frequency
for cc in cavities.element.values:
    cc.lag = 90
    cc.frequency = 0

while True:
    p_test = tw_no_rad.particle_on_co.copy()
    tracker_taper.configure_radiation(mode='mean')
    tracker_taper.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = tracker_taper.record_last_track

    eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
    print(f"Energy loss: {eloss:.3f} eV")

    if eloss < p_test.energy0[0]*rtot_eneloss:
        break

    for ii in cavities.index:
        cc = cavities.loc[ii, 'element']
        eneloss_partitioning = cavities.loc[ii, 'eneloss_partitioning']
        cc.voltage += eloss * eneloss_partitioning

    plt.plot(mon.s.T, mon.ptau.T)

i_multipoles = multipoles.index.values
delta_taper = ((mon.delta[0,:][i_multipoles+1] + mon.delta[0,:][i_multipoles]) / 2)
for nn, dd in zip(multipoles['name'].values, delta_taper):
    line[nn].knl *= (1 + dd)
    line[nn].ksl *= (1 + dd)


beta0 = p_test.beta0[0]
v_ratio = []
for icav in cavities.index:
    v_ratio.append(cavities.loc[icav, 'element'].voltage / cavities.loc[icav, 'voltage'])
    inst_phase = np.arcsin(cavities.loc[icav, 'element'].voltage / cavities.loc[icav, 'voltage'])
    freq = cavities.loc[icav, 'frequency']

    zeta = mon.zeta[0, icav]
    lag = 360.*(inst_phase / (2*np.pi) - freq*zeta/beta0/clight)

    cavities.loc[icav, 'element'].lag = lag
    cavities.loc[icav, 'element'].frequency = freq
    cavities.loc[icav, 'element'].voltage = cavities.loc[icav, 'voltage']

    import pdb; pdb.set_trace()


tw_damp = tracker.twiss(method='4d', matrix_stability_tol=0.5)
tracker_twiss = xt.Tracker(line = line, extra_headers=["#define XSUITE_SYNRAD_TWISS_MODE"])
tw_nodamp = tracker_twiss.twiss(method='4d')