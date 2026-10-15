import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Electron GaAs / Al0.3Ga0.7As superlattice, below barrier only.
HB2_2M0 = 0.0380998212  # hbar^2/(2*m0), in eV*nm^2
MASS_RATIO = 0.067
X = 0.30
DELTA_EG = 1.247 * X
V0 = 0.65 * DELTA_EG
B_LIST = [5, 10, 20, 50, 100]
A_LIST = np.linspace(1.0, 10.0, 181)
E_LOW = 1e-10
E_HIGH = V0 * (1 - 1e-10)
N_GRID = 2500


def scaled_rhs(E, a, b):
    """G(E) = F_KP(E)/cosh(beta*b), and s(E)=sech(beta*b)."""
    E = np.asarray(E)
    alpha = np.sqrt(MASS_RATIO * E / HB2_2M0)
    beta = np.sqrt(MASS_RATIO * (V0 - E) / HB2_2M0)
    # Stable sin(alpha*a)/alpha and sinh(beta*b)/beta limits.
    term = ((beta**2 - alpha**2) / (2 * beta)) * a * np.sinc(alpha * a / np.pi)
    G = np.cos(alpha * a) + term * np.tanh(beta * b)
    expneg = np.exp(-beta * b)
    s = 2 * expneg / (1 + expneg**2)
    return G, s


def find_minibands(a, b):
    """Find below-barrier allowed intervals using scaled KP root centres.

    For unresolved ultra-narrow bands, show the central energy as a line.
    """
    grid = np.linspace(E_LOW, E_HIGH, N_GRID)
    G, s = scaled_rhs(grid, a, b)
    h = np.abs(G) - s

    def hfun(e):
        g, ss = scaled_rhs(e, a, b)
        return float(abs(g) - ss)

    bands = []
    for i in np.flatnonzero(G[:-1] * G[1:] < 0):
        root = brentq(lambda e: float(scaled_rhs(e, a, b)[0]),
                      grid[i], grid[i + 1], xtol=1e-15)
        _, sr = scaled_rhs(root, a, b)
        # When band width falls below double-precision resolution,
        # plotting the centre is more honest than inventing an edge.
        if sr < 1e-11:
            bands.append((root, root))
            continue

        j = i
        while j >= 0 and h[j] <= 0:
            j -= 1
        lo = (E_LOW if j < 0 else
              brentq(hfun, grid[j], root, xtol=1e-15))

        j = i + 1
        while j < N_GRID and h[j] <= 0:
            j += 1
        hi = (E_HIGH if j >= N_GRID else
              brentq(hfun, root, grid[j], xtol=1e-15))
        bands.append((lo, hi))

    # Capture a band truncated by the barrier top even if it lacks a G zero.
    if h[-1] <= 0 and not any(hi >= E_HIGH - 1e-9 for _, hi in bands):
        j = N_GRID - 2
        while j >= 0 and h[j] <= 0:
            j -= 1
        lo = (E_LOW if j < 0 else
              brentq(hfun, grid[j], grid[j + 1], xtol=1e-15))
        bands.append((lo, E_HIGH))

    return sorted(bands)


def main():
    print(f"Electron: m* = {MASS_RATIO} m0, V0 = {V0:.6f} eV")
    for b in B_LIST:
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        all_bands = [find_minibands(a, b) for a in A_LIST]
        max_count = max(map(len, all_bands))
        for n in range(max_count):
            lower = np.array([bands[n][0] if len(bands) > n else np.nan for bands in all_bands])
            upper = np.array([bands[n][1] if len(bands) > n else np.nan for bands in all_bands])
            ax.plot(A_LIST, lower, lw=1.2, color="#174A8B", label=f"E{n+1}")
            ax.plot(A_LIST, upper, lw=1.2, color="#174A8B")
            ax.fill_between(A_LIST, lower, upper, facecolor="#E3ECF8", edgecolor="#174A8B", hatch="++", linewidth=0.0)
        ax.axhline(V0, color='#555555', ls='--', lw=1, label='Barrier top')
        ax.set(xlim=(1,10), ylim=(0,V0*1.03), xlabel='GaAs well width a (nm)',
               ylabel='Electron energy (eV)', title=f'Electron superlattice: b = {b} nm')
        ax.grid(False)
        ax.legend()
        fig.tight_layout()
        print(f"b={b:3d} nm, a=5 nm: {find_minibands(5.0,b)}")
    plt.show()


if __name__ == '__main__':
    main()
