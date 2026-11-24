import numpy as np
import matplotlib.pyplot as plt


spike_file = 'spikes_T250_A0.5.txt'
output_file = 'raster_syn_T250_A0.5.png'
NE, NI = 1000, 250

# simulation.py has already removed the initial transient and shifted the times.
time_start, time_end = 0.0, 5000.0
stim_start, stim_end = 500.0, 2500.0
stage_transient = 90.0  # Exclude this many ms at the start of each matrix window.
windows = [
    ('Before', time_start + stage_transient, stim_start),
    ('During', stim_start + stage_transient, stim_end),
    ('After', stim_end + stage_transient, time_end),
]

figsize = (17.5, 4.2)
font_size = 14
dot_size = 5
dpi = 300


def spike_syn_matrix(neuron_ids, spike_times, t0, t1):
    in_window = (neuron_ids < NE) & (t0 <= spike_times) & (spike_times < t1)
    ids = neuron_ids[in_window]
    times = spike_times[in_window]
    trains = [np.sort(times[ids == i]) for i in range(NE)]
    lengths = np.array([len(train) for train in trains])
    # For each spike, retain the shorter of its adjacent ISIs.
    # A missing neighbor at either end contributes no bound.
    train_times = np.zeros((NE, lengths.max()))
    local_isi = np.full(train_times.shape, np.inf)
    for i, train in enumerate(trains):
        n = lengths[i]
        train_times[i, :n] = train
        isi = np.diff(train)
        local_isi[i, :max(0, n - 1)] = isi
        local_isi[i, 1:n] = np.minimum(local_isi[i, 1:n], isi)

    real_spikes = np.arange(train_times.shape[1]) < lengths[:, None]
    coincidences = np.zeros((NE, NE), dtype=int)
    for j, train in enumerate(trains):
        if lengths[j] == 0:
            continue

        # Compare the nearest neighbors on both sides of each source spike.
        right = np.searchsorted(train, train_times)
        left_idx = np.clip(right - 1, 0, lengths[j] - 1)
        right_idx = np.clip(right, 0, lengths[j] - 1)
        dt_left = np.where(right > 0, np.abs(train_times - train[left_idx]), np.inf)
        dt_right = np.where(right < lengths[j], np.abs(train_times - train[right_idx]), np.inf)
        nearest = np.where(dt_left < dt_right, left_idx, right_idx)
        distance = np.minimum(dt_left, dt_right)

        # Half of the shortest of the four adjacent ISIs for this pair.
        tau = 0.5 * np.minimum(local_isi, local_isi[j, nearest])
        coincidences[:, j] = np.count_nonzero(real_spikes & (distance < tau), axis=1)

    n_spikes = lengths[:, None] + lengths[None, :]
    matrix = np.divide(
        coincidences + coincidences.T,
        n_spikes,
        out=np.zeros((NE, NE)),
        where=n_spikes > 0,
    )
    np.fill_diagonal(matrix, 1.0)
    return matrix


data = np.loadtxt(spike_file, usecols=(0, 1), ndmin=2)
spike_neuron = data[:, 0].astype(int)
spike_time = data[:, 1]

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = font_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 1.5

fig, axes = plt.subplots(
    1, 4, figsize=figsize, constrained_layout=True,
    gridspec_kw={'width_ratios': [1.8, 1, 1, 1]},
)

ax = axes[0]
e = spike_neuron < NE
ax.scatter(spike_time[e], spike_neuron[e], s=dot_size, c='r', marker='.', linewidths=0)
ax.scatter(spike_time[~e], spike_neuron[~e], s=dot_size, c='b', marker='.', linewidths=0)
ax.axvline(stim_start, color='k', linestyle='--', linewidth=1)
ax.axvline(stim_end, color='k', linestyle='--', linewidth=1)
ax.set_xlim(time_start, time_end)
ax.set_ylim(0, NE + NI)
ax.set_xticks([time_start, stim_start, stim_end, time_end])
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')

for ax, (stage, t0, t1) in zip(axes[1:], windows):
    print(f'{stage}: {t0:g}–{t1:g} ms')
    matrix = spike_syn_matrix(spike_neuron, spike_time, t0, t1)
    image = ax.imshow(matrix, origin='lower', cmap='viridis',
                      vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(stage)
    ax.set_xlabel('Neuron index')
axes[1].set_ylabel('Neuron index')
fig.colorbar(image, ax=axes[1:], label='Spike-Syn', fraction=0.035, pad=0.02)

plt.savefig(output_file, dpi=dpi)
plt.show()
