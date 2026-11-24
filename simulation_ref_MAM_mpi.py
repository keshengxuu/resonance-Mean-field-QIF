import numpy as np
import time
import os
from mpi4py import MPI


# Simulation parameters
seed = 2025
time_total = 6500.0  # ms
dt = 0.1
n_step = int(round(time_total / dt))
transient_time = 1000.0
tail_time = 500.0
analysis_end = time_total - tail_time

# Small-world network
NE = 1000
NI = 250
N = NE + NI
k = 30
p = 0.10

# EIF model parameters
Cm = 1.0
gL = 0.1
EL = -72.0
Vth = 15.0
Vre = -75.0
DeltaT = 6.0
VT = -50.0

# synaptic weights
tau_r = 5.0
tau_d = 10.0
weight_params = {
    'EE': 0.01,    # E -> E, J_EE in the paper
    'EI': 0.02,    # E -> I, J_IE in the paper
    'IE': -0.02,   # I -> E, J_EI in the paper
    'II': -0.01,   # I -> I, J_II in the paper
}
noise_min = 1.5
noise_max = 1.8


# Refractory period and perturbation amplitude at each fixed period
T_fixed_list = np.array([100.0, 200.0])  # ms
ref_list = np.arange(0.0, 60.0 + 2.0, 2.0)  # ms
A_list = np.round(np.arange(0.0, 1.0 + 0.025, 0.025), 3)

pert_params = {
    'start': 1500.0,
    'end': 3500.0,
    'neuron_range': (0, NE),
}

def build_ws_small_world(N, k, p):
    if not 0 < k < N or k % 2 or not 0.0 <= p <= 1.0:
        raise ValueError('Use an even k with 0 < k < N and 0 <= p <= 1.')

    w_ij = np.zeros((N, N), dtype=int)
    for j in range(1, k // 2 + 1):
        left = (np.arange(N) - j) % N
        right = (np.arange(N) + j) % N
        w_ij[np.arange(N), left] = 1
        w_ij[np.arange(N), right] = 1
        w_ij[left, np.arange(N)] = 1
        w_ij[right, np.arange(N)] = 1

    for i in range(N):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % N
            if np.random.rand() < p and w_ij[i].sum() < N - 1:
                w_ij[i, neighbor] = 0
                w_ij[neighbor, i] = 0
                while True:
                    new_v = np.random.randint(N)
                    if new_v != i and new_v != neighbor and w_ij[i, new_v] == 0:
                        break
                w_ij[i, new_v] = 1
                w_ij[new_v, i] = 1
    return w_ij


def input_current(N, w, J, s, current_time, pert_params):
    noise = noise_min + (noise_max - noise_min) * np.random.rand(N)
    coupling_matrix = w * J
    Isyn = np.dot(coupling_matrix.T, s)

    I_pert = np.zeros(N)
    pert_start = pert_params['start']
    pert_end = pert_params['end']
    pert_amplitude = pert_params['amplitude']
    pert_duration = pert_params['duration']
    pert_rest = pert_params['rest']
    total_cycle = pert_duration + pert_rest

    if pert_start <= current_time < pert_end:
        cycle_pos = (current_time - pert_start) % total_cycle
        if cycle_pos < pert_duration:
            start_idx, end_idx = pert_params['neuron_range']
            I_pert[start_idx:end_idx] = pert_amplitude

    return noise + Isyn + I_pert


# Example: mpiexec -n 40 python3 simulation_refA_mpi.py
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outcome')
os.makedirs(out_dir, exist_ok=True)

# Divide the scan points among MPI ranks.
tasks = [(T, A, ref) for T in T_fixed_list for ref in ref_list for A in A_list]
for T_pert, A, refractory_period in tasks[rank::size]:
    pert_params['amplitude'] = A
    pert_params['duration'] = T_pert / 2
    pert_params['rest'] = T_pert / 2
    print(f'Rank {rank}: T={T_pert:g} ms, A={A:g}, ref={refractory_period:g} ms', flush=True)

    # Build the network and run one simulation.
    np.random.seed(seed)
    w = build_ws_small_world(N, k, p)

    J = np.zeros((N, N))
    J[:NE, :NE] = weight_params['EE']
    J[:NE, NE:] = weight_params['EI']
    J[NE:, :NE] = weight_params['IE']
    J[NE:, NE:] = weight_params['II']

    tic = time.time()

    # Run the simulation.
    v_prev = np.random.uniform(Vre, Vth, N)
    next_active_step = np.zeros(N, dtype=int)
    refractory_steps = int(np.ceil(refractory_period / dt - 1e-10))
    spike_list = []

    # Sum the contributions from all past spikes using two exponential traces.
    # A new spike adds 1 to both traces, making its initial contribution zero.
    s_rise = np.zeros(N)
    s_decay = np.zeros(N)
    decay_rise = np.exp(-dt / tau_r)
    decay_decay = np.exp(-dt / tau_d)

    for i in range(n_step):
        current_time = i * dt
        s = (s_decay - s_rise) / (tau_d - tau_r)
        I = input_current(N, w, J, s, current_time, pert_params)

        # Refractory neurons stay clamped at Vre after their last reset.
        active = i >= next_active_step
        v = v_prev.copy()
        v_active = v_prev[active]
        ionic_current = (
            gL * (EL - v_active)
            + gL * DeltaT * np.exp((v_active - VT) / DeltaT)
        )
        v[active] += (dt / Cm) * (I[active] + ionic_current)

        # Euler step ends at (i + 1) * dt, the timestamp of any new spike.
        spike_mask = v >= Vth
        spike_neurons = np.where(spike_mask)[0]
        if spike_neurons.size:
            spike_time = (i + 1) * dt
            spike_list.extend((neuron, spike_time) for neuron in spike_neurons)
            next_active_step[spike_neurons] = i + 1 + refractory_steps
            v[spike_mask] = Vre
        v_prev = v

        s_rise *= decay_rise
        s_decay *= decay_decay
        s_rise[spike_mask] += 1.0
        s_decay[spike_mask] += 1.0

        if (i + 1) % max(1, n_step // 5) == 0:
            print(f'Simulation progress: {100 * (i + 1) / n_step:.0f}%')

    spikes = np.asarray(spike_list, dtype=float).reshape(-1, 2)
    print(f'Time to simulate: {time.time() - tic:.2f} sec')

    # Keep the analysis window and save zero-based neuron IDs and times in ms.
    spike_times = spikes[:, 1]
    valid_spikes = spikes[(spike_times >= transient_time) & (spike_times <= analysis_end)]
    valid_spikes[:, 1] -= transient_time
    T_tag = f'{T_pert:.4f}'.rstrip('0').rstrip('.')
    A = pert_params['amplitude']
    spike_file = os.path.join(out_dir, f'spikes_refA_T{T_tag}_A{A:g}_ref{refractory_period:g}.txt')
    np.savetxt(spike_file, valid_spikes, delimiter=' ', fmt=['%d', '%.4f'])
    print(f'Saved spike file: {spike_file}')
