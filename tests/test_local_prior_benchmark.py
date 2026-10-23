"""Synthetic benchmarks for global versus local ComBat priors.

Run: pytest -q -s test_local_priors_benchmark.py
Requires DiagnoseHarmonisation.HarmonisationFunctions on PYTHONPATH.
The comparison test reports metrics rather than assuming local must win.
"""
import numpy as np
import pandas as pd
import pytest
from DiagnoseHarmonisation import HarmonisationFunctions as hf


def make_dataset(seed=42, scenario='bimodal', n_features=120, n_per_batch=65):
    rng = np.random.default_rng(seed)
    n = 2 * n_per_batch
    batch = np.repeat(['A', 'B'], n_per_batch)
    # Covariates balanced independently of batch, preserving identifiable effects.
    age = rng.uniform(20, 75, n)
    sex = rng.integers(0, 2, n)
    age_z = (age - age.mean()) / age.std()
    coefficients = rng.normal(0.35, 0.10, n_features)
    biological = coefficients[:, None] * age_z + 0.3 * (sex - sex.mean())[None, :]
    noise = rng.normal(size=(n_features, n))
    truth = biological + noise
    centers = {
    'unimodal': [0.6],
    'bimodal': [-2.5, 2.5],
    'multimodal': [-3.5, 0.0, 3.5],
    'overlapping': [-0.5, 0.5],
    'heavy_tailed': [-2.0, 2.0],
    }

    n_groups = len(centers[scenario])
    groups = np.arange(n_features) % n_groups
    locs = np.asarray(centers[scenario])[groups]
    gamma = np.zeros((2, n_features))
    delta = np.ones((2, n_features))
    gamma[0] = locs + rng.normal(0, 0.25 if scenario != 'overlapping' else 0.7, n_features)
    if scenario == 'heavy_tailed':
        gamma[0] += rng.standard_t(3, n_features) * 0.4
    gamma[1] = -0.3 * locs + rng.normal(0, 0.25, n_features)
    delta[0] = np.exp(0.25 * (groups - groups.mean()) + rng.normal(0, 0.07, n_features))
    delta[1] = np.exp(-0.15 * (groups - groups.mean()) + rng.normal(0, 0.07, n_features))
    observed = truth.copy()
    for b in range(2):
        idx = np.flatnonzero(batch == ['A', 'B'][b])
        observed[:, idx] = biological[:, idx] + gamma[b, :, None] + np.sqrt(delta[b, :, None]) * noise[:, idx]
    return observed, truth, pd.Series(batch), pd.DataFrame({'age': age, 'sex': sex}), groups


def metrics(pred, truth):
    residual = np.asarray(pred) - truth
    return {'rmse': float(np.sqrt(np.mean(residual**2))),
            'mae': float(np.mean(np.abs(residual)))}


@pytest.mark.parametrize('scenario', ['unimodal', 'bimodal', 'multimodal', 'overlapping', 'heavy_tailed'])
@pytest.mark.parametrize('seed', [7, 21, 42])
def test_global_local_benchmark(scenario, seed):
    observed, truth, batch, mod, groups = make_dataset(seed, scenario)
    common = dict(data=observed, batch=batch, mod=mod, mean_model='ols',
                  UseEB=True, parametric=True)
    global_out = hf.combat_modular(**common, prior_mode='global')
    local_out = hf.combat_modular(
        **common, prior_mode='local',
        prior_weight_methods=['magnitude_similarity', 'variance_similarity'],
        prior_weight_opts={'similarity_floor': 0.001, 'min_effective': 5})
    global_scores = metrics(global_out['bayesdata'], truth)
    local_scores = metrics(local_out['bayesdata'], truth)
    for score in (global_scores, local_scores):
        assert all(np.isfinite(list(score.values())))
    print(f'\n{scenario:13} seed={seed:2} global={global_scores} local={local_scores} '
          f'delta_rmse={local_scores["rmse"]-global_scores["rmse"]:+.4f} '
          f'delta_mae={local_scores["mae"]-global_scores["mae"]:+.4f}')


def test_custom_block_similarity_recovers_group_specific_mean_priors():
    rng = np.random.default_rng(10)
    n_features = 60
    groups = np.arange(n_features) % 2
    gamma_true = np.where(groups == 0, -3., 3.)
    gamma_hat = gamma_true + rng.normal(0, 0.4, n_features)
    s_data = rng.normal(size=(n_features, 60))
    sim = (groups[:, None] == groups[None, :]).astype(float)
    local = hf._construct_local_priors(
        s_data=s_data, gamma_hat=gamma_hat[None, :],
        delta_hat=np.ones((1, n_features)) + rng.uniform(0, .1, (1, n_features)),
        var_pooled=np.ones(n_features), batches=[np.arange(60)],
        weight_methods=['custom'],
        weight_opts={'similarity_matrix': sim, 'similarity_floor': 1e-10,
                     'min_effective': 5})
    estimate = local['gamma_bar_local'][0]
    global_estimate = np.full(n_features, gamma_hat.mean())
    assert np.mean((estimate - gamma_true)**2) < np.mean((global_estimate - gamma_true)**2)
    assert np.all(np.isfinite(local['a_prior_local']))
    assert np.all(local['t2_local'] > 0)