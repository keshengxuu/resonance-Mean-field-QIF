import numpy as np
import matplotlib.pyplot as plt


spike_file = 'spikes_T250_A0.5.txt'
output_file = 'kuramoto_R_T250_A0.5.png'
NE = 1000

# simulation.py has already removed the initial transient and shifted the times.
t_start, t_end = 0.0, 5000.0
dt = 10.0  # Sampling interval for R(t), in ms.
stim_start, stim_end = 500.0, 2500.0
T_pert = 250.0
amplitude = 0.50

figsize = (12, 7)
font_size = 25
line_width = 2
dpi = 300


data = np.loadtxt(spike_file, usecols=(0, 1), ndmin=2)
spike_neuron = data[:, 0].astype(int)
spike_time = data[:, 1]
spike_times_by_neuron = [
    np.sort(spike_time[spike_neuron == i]) for i in range(NE)
]

time_bins = np.arange(t_start, t_end, dt)
R = np.full(len(time_bins), np.nan)

for ti, t in enumerate(time_bins):
    phases = []
    for spk_times in spike_times_by_neuron:
        j = np.searchsorted(spk_times, t, side='right')
        if 0 < j < len(spk_times):
            t_prev = spk_times[j - 1]
            t_next = spk_times[j]
            phases.append(2 * np.pi * (t - t_prev) / (t_next - t_prev))

    # Average only neurons with a spike on each side of the sample time.
    if phases:
        R[ti] = np.abs(np.mean(np.exp(1j * np.array(phases))))


# Draw the square wave on the same time axis as R(t).
wave_times = np.arange(t_start, t_end + 0.1, 0.1)
Ipert_wave = np.zeros_like(wave_times)
on = ((wave_times >= stim_start) & (wave_times < stim_end)
      & ((wave_times - stim_start) % T_pert < T_pert / 2))
Ipert_wave[on] = amplitude

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5

plt.figure(figsize=figsize)
plt.plot(time_bins, R, color='b', linewidth=line_width)
plt.xlim(t_start, t_end)
plt.ylim(0, 1.05)
plt.xlabel('Time (ms)')
plt.ylabel(r'$R(t)$')
plt.grid(True, linestyle='--', alpha=0.5)

plt.twinx()
plt.step(wave_times, Ipert_wave, where='post', color='k', linewidth=line_width)
plt.ylim(0, max(1.0, 4 * amplitude))
plt.yticks([])

plt.tight_layout()
plt.savefig(output_file, dpi=dpi)
plt.show()
