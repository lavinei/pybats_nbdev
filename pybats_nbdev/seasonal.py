# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/04_seasonal.ipynb (unless otherwise specified).

__all__ = ['seascomp', 'createFourierToSeasonalL', 'fourierToSeasonal', 'fourierToSeasonalFxnl']

# Internal Cell
#exporti
import numpy as np
from .forecast import forecast_aR, forecast_R_cov

# Cell
def seascomp(period, harmComponents):
    p = len(harmComponents)
    n = 2*p
    F = np.zeros([n, 1])
    F[0:n:2] = 1
    G = np.zeros([n, n])

    for j in range(p):
        c = np.cos(2*np.pi*harmComponents[j]/period)
        s = np.sin(2*np.pi*harmComponents[j]/period)
        idx = 2*j
        G[idx:(idx+2), idx:(idx+2)] = np.array([[c, s],[-s, c]])

    return [F, G]

# Cell
def createFourierToSeasonalL(period, harmComponents, Fseas, Gseas):
    p = len(harmComponents)
    L = np.zeros([period, 2*p])
    L[0,:] = Fseas.reshape(-1)
    for i in range(1, period):
        L[i,:] = L[i-1,:] @ Gseas

    return L

# Cell
def fourierToSeasonal(mod):
    phi = mod.L @ mod.m[mod.iseas]
    var = mod.L @ mod.C[np.ix_(mod.iseas, mod.iseas)] @ mod.L.T
    return phi, var

# Cell
def fourierToSeasonalFxnl(L, m, C, iseas):
    phi = L @ m[iseas]
    var = L @ C[np.ix_(iseas, iseas)] @ L.T
    return phi, var

# Internal Cell
def get_seasonal_effect_fxnl(L, m, C, iseas):
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    return phi[0], var[0, 0]

# Internal Cell
def sample_seasonal_effect_fxnl(L, m, C, iseas, delVar, n, nsamps):
    phi_samps = np.zeros([nsamps])
    phi, var = fourierToSeasonalFxnl(L, m, C, iseas)
    phi_samps[:] = phi[0] + np.sqrt(var[0,0])*np.random.standard_t(delVar*n, size = [nsamps])
    return phi_samps

# Internal Cell
def forecast_weekly_seasonal_factor(mod, k, sample = False, nsamps = 1):
    a, R = forecast_aR(mod, k)

    idx = np.where(np.array(mod.seasPeriods) == 7)[0][0]

    if sample:
        return sample_seasonal_effect_fxnl(mod.L[idx], a, R, mod.iseas[idx], mod.delVar, mod.n, nsamps)
    else:
        return get_seasonal_effect_fxnl(mod.L[idx], a, R, mod.iseas[idx])

# Internal Cell
def forecast_path_weekly_seasonal_factor(mod, k, today, period):
    phi_mu = [np.zeros([period]) for h in range(k)]
    phi_sigma = [np.zeros([period, period]) for h in range(k)]
    phi_psi = [np.zeros([period, period, h]) for h in range(1, k)]

    idx = np.where(np.array(mod.seasPeriods) == 7)[0][0]
    L = mod.L[idx]
    iseas = mod.iseas[idx]

    for h in range(k):

        # Get the marginal a, R
        a, R = forecast_aR(mod, h + 1)

        m, v = get_seasonal_effect_fxnl(L, a, R, iseas)
        day = (today + h) % period
        phi_mu[h][day] = m
        phi_sigma[h][day, day] = v
        # phi_mu[h], phi_sigma[h] = get_latent_factor_fxnl_old((today + h) % period, mod.L, a, R, mod.iseas, mod.seasPeriods[0])

        # Find covariances with previous latent factor values
        for j in range(h):
            # Covariance matrix between the state vector at times j, i, i > j
            day_j = (today + j) % period
            cov_jh = forecast_R_cov(mod, j, h)[np.ix_(iseas, iseas)]
            phi_psi[h-1][day, day_j, j] = phi_psi[h-1][day_j, day, j] = (L @ cov_jh @ L.T)[day, day_j]
            # cov_ij = (np.linalg.matrix_power(mod.G, h-j) @ Rlist[j])[np.ix_(mod.iseas, mod.iseas)]

    return phi_mu, phi_sigma, phi_psi