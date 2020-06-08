import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

def load_config(result_dir):
    config_file = os.path.join(result_dir, 'config.pkl')
    with open(config_file, 'rb') as f:
        return pickle.load(f)

def load_result(result_dir):
    history_file = os.path.join(result_dir, 'history.csv')
    return pd.read_csv(history_file)

def compute_mean_time(r):
    return r[r.epoch>0].time.mean()

def get_num_samples(config, ranks):
    dconf = config['data']
    n = dconf['n_train'] + dconf['n_valid']
    if not dconf['shard']:
        n *= ranks
    return n

def get_scaling_results(path_pattern, ranks):
    """
    Loops over ranks with specified file path pattern and computes scaling metrics.
    Returns results in a dataframe.
    """
    configs, results = [], []
    for r in ranks:
        result_dir = path_pattern % r
        configs.append(load_config(result_dir))
        results.append(load_result(result_dir).assign(ranks=r))
    samples = np.array([get_num_samples(c,r) for (c,r) in zip(configs, ranks)])    
    times = np.array([compute_mean_time(r) for r in results])
    throughputs = samples / times
    ideal = ranks * throughputs[0]/4 # Change to the GPU/node
    eff = throughputs / ideal
    return pd.DataFrame(dict(ranks=ranks, samples=samples,
                             times=times, throughputs=throughputs,
                             ideal=ideal, eff=eff))




results_cgpu = get_scaling_results(
    os.path.expandvars('/gpfs/alpine/world-shared/stf011/atsaris/cosmoflow_output_2020_rk%i'),
    ranks=np.array([4, 8, 16, 32, 64]))
    #ranks=np.array([1, 2, 4, 8, 16, 32, 128]))

# Summary table
print(results_cgpu)
#results_cgpu.merge(results_cgpu_dummy, on='ranks', suffixes=(None,'_dummy'))

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(8,8),
                               gridspec_kw=dict(height_ratios=[.8, .2], hspace=0))

ax0.plot(results_cgpu.ranks, results_cgpu.throughputs, 'o-', ms=8, label='Real data')
#ax0.plot(results_cgpu_dummy.ranks, results_cgpu_dummy.throughputs, '^-', ms=8, label='Dummy data')
ax0.plot(results_cgpu.ranks, results_cgpu.ideal, '--', label='Ideal')
ax0.set_ylabel('Training throughput [samples/s]')
ax0.legend(loc=0)
ax0.grid()

# Scaling efficiency
ax1.plot(results_cgpu.ranks, results_cgpu.eff, 'o-', ms=8)
ax1.set_xlabel('Number of workers')
ax1.set_ylabel('Efficiency')
ax1.set_ylim(bottom=0.5)
ax1.grid()

plt.tight_layout()
plt.show()
