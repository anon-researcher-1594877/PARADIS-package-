"""Stochastic dispersal components.

Two complementary modes handle low-density individuals that are below the
deterministic presence threshold:

1. **Stochastic local walk** – an individual performs a discrete random walk
   on the habitat graph, stopping after a geometrically distributed number of
   steps.
2. **Long-distance dispersal (LDD)** – with probability ``eta`` an individual
   is instead transported to a random location drawn from a log-normal kernel.

:func:`stochastic_dispersal_step`
    Vectorised implementation of both modes.
:func:`lognormal_2d_kernel`
    Build a 2-D log-normal dispersal kernel (for deterministic LDD).
:func:`apply_kernel`
    Convolve a density map with a dispersal kernel.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def stochastic_dispersal_step(
    distrib: np.ndarray,
    hs: np.ndarray,
    ewalk: float = 20.0,
    n: float = 400.0,
    r: float = 0.17,
    eta: float = 0.2,
    ldd_mu: float = 2.75,
    ldd_sigma2: float = 0.5,
    ldd_attempts: int = 12,
) -> np.ndarray:
    """Apply a stochastic dispersal step to a density map.

    Parameters
    ----------
    distrib:
        2-D density map.  Each non-zero pixel is treated as an individual
        unit to be dispersed.
    hs:
        2-D habitat-suitability map (same shape as *distrib*).
    ewalk:
        Expected number of random-walk steps.
    n:
        Risk-avoidance exponent (higher → more selective movement).
    r:
        Risk-scaling exponent (survival = HS^r per step).
    eta:
        Proportion of LDD events.
    ldd_mu, ldd_sigma2:
        Log-normal parameters for LDD distance distribution
        (``distance = exp(ldd_mu + sqrt(ldd_sigma2) * Z)`` where ``Z ~ N(0,1)``).
    ldd_attempts:
        Number of direction trials for LDD before giving up.

    Returns
    -------
    numpy.ndarray
        Updated density map (same shape).
    """
    H, W = distrib.shape
    new_distrib = np.zeros_like(distrib)
    step_prob = ewalk / (1.0 + ewalk)
    rn_alpha = r * n
    sigma = np.sqrt(ldd_sigma2)

    x0s, y0s = np.where(distrib > 0)
    nb_ldd = 0

    for k in tqdm(range(len(x0s)), desc="Stochastic dispersal", leave=False):
        x0, y0 = x0s[k], y0s[k]
        carried = distrib[x0, y0]

        # Long-distance dispersal
        if np.random.random() < eta:
            dist = np.exp(ldd_mu + sigma * np.random.randn())
            for _ in range(ldd_attempts):
                theta = 2.0 * np.pi * np.random.random()
                xn = int(x0 + dist * np.cos(theta))
                yn = int(y0 + dist * np.sin(theta))
                if 0 <= xn < H and 0 <= yn < W and hs[xn, yn] != 0.0:
                    new_distrib[xn, yn] += carried
                    nb_ldd += 1
                    break
            continue

        # Local stochastic walk
        xc, yc = x0, y0
        while True:
            if np.random.random() < step_prob:
                # Choose direction proportionally to hs^(r * n)
                w0 = hs[xc + 1, yc] ** rn_alpha if xc + 1 < H else 0.0
                w1 = hs[xc - 1, yc] ** rn_alpha if xc - 1 >= 0 else 0.0
                w2 = hs[xc, yc + 1] ** rn_alpha if yc + 1 < W else 0.0
                w3 = hs[xc, yc - 1] ** rn_alpha if yc - 1 >= 0 else 0.0
                ws = np.array([w0, w1, w2, w3], dtype=float)
                total = ws.sum()
                if total == 0:
                    break
                ws /= total
                direction = np.random.choice(4, p=ws)
                moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                dx, dy = moves[direction]
                xc += dx
                yc += dy
            else:
                new_distrib[xc, yc] += carried
                break

            # Survival check
            if np.random.random() > hs[xc, yc] ** r:
                break

    return new_distrib


def lognormal_2d_kernel(
    mu: float,
    sigma: float,
    size: int = 201,
) -> np.ndarray:
    """Build a 2-D log-normal dispersal kernel.

    Parameters
    ----------
    mu, sigma:
        Log-normal parameters (``E[ln(distance)] = mu``,
        ``Std[ln(distance)] = sigma``).
    size:
        Kernel side length (must be odd).

    Returns
    -------
    numpy.ndarray
        Normalised 2-D kernel of shape ``(size, size)``.
    """
    half = size // 2
    y, x = np.mgrid[-half: half + 1, -half: half + 1]
    dist = np.sqrt(x ** 2 + y ** 2).astype(float)
    dist[dist == 0] = 1e-10
    kernel = np.exp(
        -((np.log(dist) - mu) ** 2) / (2.0 * sigma ** 2)
    ) / (dist * sigma * np.sqrt(2.0 * np.pi))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def apply_kernel(
    density: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Convolve a density map with a dispersal kernel.

    Parameters
    ----------
    density:
        2-D density map.
    kernel:
        2-D kernel (must be smaller than *density*).

    Returns
    -------
    numpy.ndarray
        Convolved density map (same shape as *density*).
    """
    from scipy.signal import fftconvolve

    result = fftconvolve(density, kernel, mode="same")
    result[result < 0] = 0.0
    return result.astype(np.float32)
