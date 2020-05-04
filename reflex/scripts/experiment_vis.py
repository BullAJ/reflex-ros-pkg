import numpy as np
import matplotlib.pyplot as plotter
from readings import _FILTER_CUTOFF, _NUM_SAMPLES, _DIFF, _NUM_SENSORS

filter_cutoff = int(_FILTER_CUTOFF)
num_samples = _NUM_SAMPLES

fig, (ax1, ax2) = plotter.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)

dt = 0.02  # 50hz
t = np.arange(0, 24, dt)

diffstr = '_diff' if _DIFF else ''
clean_data = None
with open('unfiltered_samples{:s}.out'.format(diffstr)) as cleanf:
    clean_data = np.loadtxt(cleanf)

filtered_data = None
with open('{:d}hz_{:d}samples{:s}.out'.format(filter_cutoff, num_samples, diffstr)) as filteredf:
    filtered_data = np.loadtxt(filteredf)


# Only get first finger
finger_clean    = np.transpose(clean_data[:,0:_NUM_SENSORS])
finger_filtered = np.transpose(filtered_data[:,0:_NUM_SENSORS])

for i in range(_NUM_SENSORS):
    ax1.plot(t, finger_clean[i])
#     ax2.plot(t, finger_filtered[i])

# ax1.plot(t, np.sum(finger_clean, axis=0))
ax2.plot(t, np.sum(finger_filtered, axis=0))

ax1.set_xlabel('Time')
ax1.set_ylabel('Pressure')
ax2.set_xlabel('Time')
ax2.set_ylabel('{:.1f} Hz, {:d} Samples Filter'.format(_FILTER_CUTOFF, num_samples))

plotter.savefig('/home/aj/Documents/CS4940/{:d}hz_{:d}samples{:s}_comparison_sum.png'.format(filter_cutoff, num_samples, diffstr))
# plotter.savefig('{:d}hz_{:d}samples_comparison_sum.png'.format(filter_cutoff, num_samples))

plotter.show()
