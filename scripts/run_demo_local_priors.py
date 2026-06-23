#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
# Ensure repo root is first on sys.path so local package is imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DiagnoseHarmonisation.HarmonisationFunctions import combat_modular

np.random.seed(0)

n_features = 50
n_samples = 120
coords = np.linspace(0,1,n_features)[:,None]
true_feature_amp = np.sin(2*np.pi*coords[:,0])*0.5 + 1.0
age = np.linspace(0, 2*np.pi, n_samples)
signal = np.outer(true_feature_amp, np.sin(age))

batch = np.random.choice(['A','B'], size=n_samples)
batch_effect_profile = np.cos(4*np.pi*coords[:,0]) * 0.6
batch_effect = np.where(batch == 'A', batch_effect_profile[:,None], -batch_effect_profile[:,None])

data = signal + batch_effect + 0.2 * np.random.randn(n_features, n_samples)
data_df = pd.DataFrame(data)
mod = pd.DataFrame({'age': age})

print('Running global prior harmonisation...')
out_global = combat_modular(data=data_df, batch=pd.Series(batch), mod=mod, mean_model='gam', prior_mode='global', return_priors=True)
print('Running local prior harmonisation...')
out_local = combat_modular(
    data=data_df,
    batch=pd.Series(batch),
    mod=mod,
    mean_model='gam',
    prior_mode='local',
    prior_weight_methods=['spatial_proximity','directional_bias'],
    prior_weight_opts={'feature_coords': coords, 'method_weights': {'spatial_proximity':0.4,'directional_bias':0.6}, 'directional':{'dir_sigma':0.1,'dir_power':1.2}},
    return_priors=True,
)

# Extract bayesdata
bayes_global = out_global['bayesdata'] if isinstance(out_global, dict) else out_global
bayes_local = out_local['bayesdata'] if isinstance(out_local, dict) else out_local
arr_g = np.asarray(bayes_global)
arr_l = np.asarray(bayes_local)
mse_global = np.mean((arr_g - signal)**2)
mse_local = np.mean((arr_l - signal)**2)
print(f'MSE global: {mse_global:.6f}')
print(f'MSE local:  {mse_local:.6f}')

# Save weight matrix plot
local_priors = out_local['priors'].get('local_priors', None) if isinstance(out_local, dict) else None
if local_priors is not None:
    W0 = local_priors['weights'][0]
    plt.figure(figsize=(6,6))
    plt.imshow(W0, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Local weight matrix (batch 0)')
    outpath='notebooks/demo_local_weights_batch0.png'
    plt.savefig(outpath, bbox_inches='tight')
    print('Saved weight matrix to', outpath)
else:
    print('No local_priors found to save weight matrix.')

# Save results to JSON
try:
    import json
    results = {'mse_global': float(mse_global), 'mse_local': float(mse_local)}
    with open('notebooks/demo_local_results.json','w') as f:
        json.dump(results, f)
    print('Saved results to notebooks/demo_local_results.json')
except Exception as e:
    print('Failed saving results:', e)

# Summary conclusion
if mse_local < mse_global:
    print('Local priors improved MSE.')
else:
    print('Local priors did NOT improve MSE in this run.')
