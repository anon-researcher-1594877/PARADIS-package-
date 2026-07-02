"""Population growth model.

The growth model uses a discrete logistic-type recursion::

    U_{t+1} = U_t + (1 - a) * U_t * (1 - U_t / K)

where ``a = 0.05^{1/Tg}`` is the linear growth coefficient and ``K`` is
the per-pixel carrying capacity inferred from the habitat-suitability map.

Key functions
-------------
:func:`growth_coefficient`
    Compute the scalar growth coefficient ``a`` from the characteristic
    growth time ``Tg``.
:func:`carrying_capacity_from_hs`
    Compute per-pixel carrying capacities using a fitted logistic
    relationship between HS and relative abundance.
:func:`equilibrium_distribution`
    Iterate the growth–dispersal loop to approximate the stationary
    distribution.
:func:`growth_step`
    Apply a single growth step to a 2-D abundance map.

Classes
-------
:class:`GrowthModel`
    Object-oriented wrapper around the growth functions.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from paradis._device import device


def growth_coefficient(tgrowth: float) -> float:
    """Return the linear growth coefficient ``a = 0.05^{1/Tg}``.

    Parameters
    ----------
    tgrowth:
        Characteristic growth time – number of time steps required for a
        population starting at near-zero density to reach 95 % of carrying
        capacity.

    Returns
    -------
    float
    """
    return 0.05 ** (1.0 / tgrowth)


def _logistic(x: np.ndarray | torch.Tensor, L: float, k: float, x0: float):
    """Logistic curve shifted so that ``f(0) = 0``.

    ``f(x) = L / (1 + exp(k*(x - x0))) - L / (1 + exp(k*(0 - x0)))``

    Parameters
    ----------
    x:
        Input values (HS scores in ``[0, 1]``).
    L, k, x0:
        Shape, slope, and inflection-point parameters.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Carrying-capacity values with the same type as *x*.
    """
    def g(t):
        if isinstance(t, torch.Tensor):
            z = torch.clamp(k * (t - x0), -700, 700)
            return L / (1.0 + torch.exp(z))
        else:
            z = np.clip(k * (t - x0), -700, 700)
            return L / (1.0 + np.exp(z))

    return g(x) - g(0)


def carrying_capacity_from_hs(
    hs: np.ndarray | torch.Tensor,
    tgrowth: float,
    L: float,
    k: float,
    x0: float,
) -> tuple:
    """Compute growth parameters from a habitat-suitability map.

    Parameters
    ----------
    hs:
        2-D or flat array of HS values in ``[0, 1]``.
    tgrowth:
        Characteristic growth time.
    L, k, x0:
        Logistic parameters for the K(HS) relationship.

    Returns
    -------
    a : float
        Linear growth coefficient.
    K_is : torch.Tensor
        Per-pixel carrying capacities (flat).
    vecb : torch.Tensor
        Per-pixel growth intercepts ``K_i * (1 - a)``.
    """
    if not isinstance(hs, torch.Tensor):
        hs = torch.tensor(hs, dtype=torch.float32, device="cpu")
    hs_flat = hs.flatten().float()
    K_is = _logistic(hs_flat, L, k, x0)
    K_is = torch.tensor(np.array(K_is), dtype=torch.float32) if isinstance(K_is, np.ndarray) else K_is.float()
    a = growth_coefficient(tgrowth)
    vecb = K_is * (1.0 - a)
    return a, K_is, vecb


def equilibrium_distribution(
    init_distrib: torch.Tensor,
    kernel: torch.Tensor,
    linear_growth: float | torch.Tensor,
    n_iter: int = 10,
    plot: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """Approximate the stationary distribution by iterating dispersal + growth.

    Parameters
    ----------
    init_distrib:
        Flat initial distribution (length ``N``), used also as carrying
        capacity (``K = init_distrib``).
    kernel:
        ``(N, N)`` dispersal transition matrix.
    linear_growth:
        Scalar growth coefficient ``a``.
    n_iter:
        Number of iterations (default 10).
    plot:
        If ``True``, plot the distribution at each iteration.
    verbose:
        If ``True``, print intermediate statistics.

    Returns
    -------
    torch.Tensor
        Approximate stationary distribution (length ``N``).
    """
    carrying_cap = init_distrib.to(device)
    Un = init_distrib.to(device)
    list_changes = []

    for epoch in range(n_iter):
        prev = Un.clone().detach()

        if plot:
            size = int(Un.shape[0] ** 0.5)
            plt.figure()
            plt.imshow(Un.detach().cpu().numpy().reshape(size, size), cmap="viridis")
            plt.title(f"Distribution – iteration {epoch}")
            plt.colorbar()
            plt.show()

        Un = Un @ kernel  # dispersal step

        # Clamp carrying_cap away from 0 to prevent Un/0 = inf.
        # nan_to_num only fixes NaN; -inf from positive/0 passes through and
        # causes NaN gradients in the backward pass.  The clamp value (1e-7) is
        # far below any physically meaningful carrying capacity.
        safe_cap = carrying_cap.clamp(min=1e-7)
        growth = (1.0 - linear_growth) * Un * (1.0 - Un / safe_cap)
        growth = torch.nan_to_num(growth, nan=0.0, posinf=0.0, neginf=0.0)

        dispersers = torch.clamp(growth, min=0.0)
        negative_growth = growth - dispersers
        Un = Un + negative_growth
        Un = torch.maximum(Un, torch.zeros(1, device=Un.device))
        Un = Un + dispersers @ kernel
        Un = torch.clamp(Un, 0.0, 1.0)

        change = torch.nansum(torch.abs(Un - prev)).item()
        list_changes.append(change)
        if verbose:
            print(f"  iter {epoch}: change = {change:.6f}")

    if plot:
        plt.figure()
        plt.plot(list_changes)
        plt.xlabel("Iteration")
        plt.ylabel("Total change")
        plt.title("Convergence to equilibrium")
        plt.show()

    return Un


def growth_step(
    distrib: torch.Tensor,
    linear_growth: float | torch.Tensor,
    K_is: torch.Tensor,
    tested_xs: list,
    tested_ys: list,
    window_half_size: int,
    threshold_abundance: float,
    blacklisted_points: list | None = None,
    breeding_ground: torch.Tensor | None = None,
    plot: bool = False,
) -> tuple[torch.Tensor, list]:
    """Apply one growth step to a full 2-D abundance map.

    Parameters
    ----------
    distrib:
        2-D abundance map ``(H, W)``.
    linear_growth:
        Scalar growth coefficient.
    K_is:
        Flat carrying-capacity vector (length ``H * W``).
    tested_xs, tested_ys:
        Coordinates of recently evaluated dispersal centres.
    window_half_size:
        Half-width of the dispersal evaluation window (``mw / 2``).
    threshold_abundance:
        Relative-abundance value corresponding to species presence.
    blacklisted_points:
        List of ``(x, y)`` tuples already near carrying capacity.
    breeding_ground:
        Optional 2-D binary mask – growth only where this is 1.
    plot:
        If ``True``, visualise the growth contribution.

    Returns
    -------
    new_distrib : torch.Tensor
        Updated 2-D abundance map.
    blacklisted_points : list
        Updated blacklist.
    """
    if blacklisted_points is None:
        blacklisted_points = []

    m, n = distrib.shape
    K_raster = K_is.reshape(m, n)
    un = distrib.flatten().clone().detach().to(device)

    if breeding_ground is None:
        bg = torch.ones((m, n), device=device, dtype=torch.float32).flatten()
    else:
        if not isinstance(breeding_ground, torch.Tensor):
            breeding_ground = torch.tensor(breeding_ground, dtype=torch.float32, device=device)
        bg = breeding_ground.flatten().clone().detach().to(device)

    growth = (1.0 - linear_growth) * un * (1.0 - un / K_is.to(device))
    growth = torch.nan_to_num(growth, nan=0.0)
    growth = growth * bg

    new_un = un + growth
    new_un = torch.clamp(new_un, 0.0, 1.0)
    new_un = torch.nan_to_num(new_un, nan=0.0)
    new_distrib = new_un.reshape(m, n)

    if plot:
        plt.figure()
        ratio = (growth / (new_un + 1e-10)).reshape(m, n)
        plt.imshow(ratio.cpu().numpy(), cmap="plasma", vmin=-1, vmax=1)
        plt.colorbar(label="Growth / (Growth + U_t)")
        plt.title("Growth contribution")
        plt.show()

    return new_distrib, blacklisted_points


# ---------------------------------------------------------------------------
# Object-oriented wrapper
# ---------------------------------------------------------------------------

class GrowthModel:
    """Encapsulates the logistic growth model for a species.

    Parameters
    ----------
    tgrowth:
        Characteristic growth time (years / time steps).
    L, k, x0:
        Logistic parameters describing ``K(HS)``.

    Examples
    --------
    >>> import numpy as np
    >>> from paradis.core.growth import GrowthModel
    >>> gm = GrowthModel(tgrowth=7.5, L=0.05, k=5.0, x0=0.5)
    >>> hs = np.random.rand(50, 50).astype("float32")
    >>> a, K_is, vecb = gm.carrying_capacity(hs)
    """

    def __init__(
        self,
        tgrowth: float,
        L: float,
        k: float,
        x0: float,
    ) -> None:
        self.tgrowth = tgrowth
        self.L = L
        self.k = k
        self.x0 = x0
        self._a = growth_coefficient(tgrowth)

    @property
    def a(self) -> float:
        """Scalar linear growth coefficient."""
        return self._a

    def carrying_capacity(self, hs: np.ndarray | torch.Tensor) -> tuple:
        """Compute carrying-capacity parameters for *hs*.

        Returns
        -------
        a, K_is, vecb
            See :func:`carrying_capacity_from_hs`.
        """
        return carrying_capacity_from_hs(hs, self.tgrowth, self.L, self.k, self.x0)

    def __repr__(self) -> str:
        return (
            f"GrowthModel(tgrowth={self.tgrowth}, "
            f"L={self.L}, k={self.k}, x0={self.x0})"
        )
