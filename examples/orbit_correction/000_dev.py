import xtrack as xt
import numpy as np

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
tt = line.get_table()
line.twiss_default['co_search_at'] = 'ip7'

tw0 = line.twiss4d()

# line['mq.14r7.b1'].shift_x=0.5e-3
# line['mq.14r2.b1'].shift_y=0.5e-3

tw1 = line.twiss4d()

# Select monitors by names (starting by "bpm." and not ending by "_entry" or "_exit")
tt_monitors = tt.rows['bpm\..*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
h_monitor_names = tt_monitors.name
v_monitor_names = tt_monitors.name

# Select h correctors by names (starting by "mcb.", containing "h.", and ending by ".b1")
tt_h_correctors = tt.rows['mcb.*'].rows['.*h\..*'].rows['.*\.b1']
# Select v correctors by names (starting by "mcb.", containing "v.", and ending by ".b1")
tt_v_correctors = tt.rows['mcb.*'].rows['.*v\..*'].rows['.*\.b1']


h_corrector_names = tt_h_correctors.name
v_corrector_names = tt_v_correctors.name

betx_monitors = tw1.rows[h_monitor_names].betx
bety_monitors = tw1.rows[v_monitor_names].bety

betx_correctors = tw1.rows[h_corrector_names].betx
bety_correctors = tw1.rows[v_corrector_names].bety

mux_monitor = tw1.rows[h_monitor_names].mux
muy_monitor = tw1.rows[v_monitor_names].muy

mux_correctors = tw1.rows[h_corrector_names].mux
muy_correctors = tw1.rows[v_corrector_names].muy

n_h_monitors = len(h_monitor_names)
n_v_monitors = len(v_monitor_names)

n_hcorrectors = len(h_corrector_names)
n_vcorrectors = len(v_corrector_names)

n_hmonitors = len(h_monitor_names)
n_vmonitors = len(v_monitor_names)

qx = tw1.qx
qy = tw1.qy

from numpy.matlib import repmat

# Slide 28
# https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf

bet_prod_x = np.atleast_2d(betx_monitors).T @ np.atleast_2d(betx_correctors)
bet_prod_y = np.atleast_2d(bety_monitors).T @ np.atleast_2d(bety_correctors)

mux_diff = repmat(mux_monitor, n_hcorrectors, 1).T - repmat(mux_correctors, n_hmonitors, 1)
muy_diff = repmat(muy_monitor, n_vcorrectors, 1).T - repmat(muy_correctors, n_vmonitors, 1)

response_matrix_x = np.sqrt(bet_prod_x) / 2 / np.sin(np.pi * qx) * np.cos(np.pi * qx - 2*np.pi*np.abs(mux_diff))
response_matrix_y = np.sqrt(bet_prod_y) / 2 / np.sin(np.pi * qy) * np.cos(np.pi * qy - 2*np.pi*np.abs(muy_diff))

name_h_kick = 'mcbh.15r7.b1'
name_y_kick = 'mcbv.14r7.b1'

theta_x = 1e-5
theta_y = 2e-5
i_h_kick = np.where(h_corrector_names == name_h_kick)[0][0]
i_v_kick = np.where(v_corrector_names == name_y_kick)[0][0]

line[name_h_kick].knl[0] = -theta_x
line[name_y_kick].ksl[0] = theta_y

tw2 = line.twiss4d(only_orbit=True)

kick_vect_x = np.zeros(n_hcorrectors)
kick_vect_x[i_h_kick] = theta_x

kick_vect_y = np.zeros(n_vcorrectors)
kick_vect_y[i_v_kick] = theta_y

x_res = response_matrix_x @ kick_vect_x
y_res = response_matrix_y @ kick_vect_y

x_meas = tw2.rows[h_monitor_names].x
y_meas = tw2.rows[v_monitor_names].y
x_s_meas = tw2.rows[h_monitor_names].s
y_s_meas = tw2.rows[v_monitor_names].s

class MeasOrbitH(xt.Action):
    def __init__(self, line, h_monitor_names):
        self.line = line
        self.h_monitor_names = h_monitor_names
        self.n_monitors = len(h_monitor_names)

    def run(self):
        tw = self.line.twiss4d(only_orbit=True)
        x = tw.rows[self.h_monitor_names].x
        out = {f'x_at_'+name: x[i] for i, name in enumerate(self.h_monitor_names)}
        return out
meas_orbit_h = MeasOrbitH(line, h_monitor_names)

tol=1e-8

h_correction_knobs = []
for nn_kick in h_corrector_names:
    corr_knob_name = f'orbit_corr_{nn_kick}'
    assert hasattr(line[nn_kick], 'knl')
    line.vars[corr_knob_name] = 0
    line.element_refs[nn_kick].knl[0] += line.vars[f'orbit_corr_{nn_kick}']
    h_correction_knobs.append(corr_knob_name)

opt = line.match(
    solve=False,
    targets=[
        meas_orbit_h.target(f'x_at_{name}', value=0., tol=tol)
            for name in h_monitor_names],
    vary=xt.VaryList(h_correction_knobs, step=1e-8))

def _get_jacobian(x, **kwargs):
    return -response_matrix_x
opt._err.get_jacobian = _get_jacobian

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(tt_monitors.s, x_res, '.', label='Response')
plt.plot(tw2.s, tw2.x)
plt.plot(x_s_meas, x_meas, 'x', label='Measurement')
plt.ylabel('x')
plt.grid(True)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(tt_monitors.s, y_res, '.', label='Response')
plt.plot(tw2.s, tw2.y)
plt.plot(y_s_meas, y_meas, 'x', label='Measurement')
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.show()

