"""
models.py -- every number the animations show comes from here. Pure numpy, no ManimGX,
so `python3 models.py` checks all of the math in a second.

    Part 0  linear algebra warm-up, the composition operator on a grid, cosine concentration
    Part 1  the exact Koopman map, EDMD, the pendulum
    Part 2  hyperdimensional computing (bipolar vectors, bind / bundle / permute)
"""
import numpy as np


# ======================================================================= Part 0
def warmup_matrix(lam, mu):
    """A = V diag(lam, mu) V^-1 with eigenvectors v1 = (1, 0), v2 = (1, -1)."""
    V = np.array([[1.0, 1.0], [0.0, -1.0]])
    return V @ np.diag([lam, mu]) @ np.linalg.inv(V), V


def map1d(x, eps=0.25):
    """A monotone nonlinear map of [-1, 1] onto itself: fixed points -1, 0, 1."""
    return x - eps * np.sin(np.pi * x)


def composition_matrix(grid, F):
    """(P g)_i = g(F(x_i)) for g sampled on `grid`, by linear interpolation. This is the
    Koopman operator g -> g o F written as an N x N matrix."""
    n = len(grid)
    P = np.zeros((n, n))
    for i, fx in enumerate(F(grid)):
        j = int(np.clip(np.searchsorted(grid, fx) - 1, 0, n - 2))
        w = (fx - grid[j]) / (grid[j + 1] - grid[j])
        P[i, j], P[i, j + 1] = 1 - w, w
    return P


def cosine_samples(D, n, rng):
    """cos(u, v) for n pairs of independent uniform vectors in {-1, +1}^D.
    u.v = D - 2 * (#disagreements) and #disagreements ~ Binomial(D, 1/2)."""
    return 1.0 - 2.0 * rng.binomial(D, 0.5, size=n) / D


def cosine_histogram(D, n, rng, width=1 / 32):
    """Fraction of pairs per bin; bins centred on multiples of `width` in [-1, 1]."""
    k = int(round(1 / width))
    edges = (np.arange(-k, k + 2) - 0.5) * width
    counts, _ = np.histogram(cosine_samples(D, n, rng), bins=edges)
    return (np.arange(-k, k + 1) * width), counts / n


# ======================================================================= Part 1
def exact_map(x, y, lam, mu):
    """x' = lam x,  y' = mu y + (lam^2 - mu) x^2   (nonlinear, but Koopman-exact)."""
    return lam * x, mu * y + (lam ** 2 - mu) * x ** 2


def lift(x, y):
    """z = (x, y, x^2): the observables that span a K-invariant subspace."""
    return np.array([x, y, x ** 2])


def lift_matrix(lam, mu):
    """z_{n+1} = A z_n for z = lift(x, y)."""
    return np.array([[lam, 0, 0], [0, mu, lam ** 2 - mu], [0, 0, lam ** 2]])


def koopman_coeff_matrix(lam, mu):
    """K on coefficients c of g = c_x x + c_y y + c_{x2} x^2 (the transpose of A)."""
    return lift_matrix(lam, mu).T


def eigenfunctions(lam, mu):
    """(eigenvalue, phi, tex) for the three eigenfunctions in span{x, y, x^2}."""
    return [(lam, lambda x, y: x, r"x"),
            (mu, lambda x, y: y - x ** 2, r"y-x^2"),
            (lam ** 2, lambda x, y: x ** 2, r"x^2")]


def orbit(x0, y0, lam, mu, n):
    pts = [(x0, y0)]
    for _ in range(n):
        pts.append(exact_map(*pts[-1], lam, mu))
    return np.array(pts)


def edmd(psi, X, Y):
    """Least-squares K with psi(Y) ~ psi(X) K (rows are samples)."""
    PX = np.array([psi(*p) for p in X])
    PY = np.array([psi(*p) for p in Y])
    return np.linalg.lstsq(PX, PY, rcond=None)[0]


def edmd_predict(psi, K, p0, n):
    """Roll the linear model forward in observable space, read back (x, y)."""
    z = np.array(psi(*p0), dtype=float)
    out = [z[:2].copy()]
    for _ in range(n):
        z = z @ K
        out.append(z[:2].copy())
    return np.array(out)


def psi_dmd(x, y):
    return np.array([x, y])


def psi_exact(x, y):
    return np.array([x, y, x ** 2])


def edmd_data(lam, mu, n, rng, box=((-2, 2), (-1, 4))):
    X = np.column_stack([rng.uniform(*box[0], n), rng.uniform(*box[1], n)])
    Y = np.array([exact_map(x, y, lam, mu) for x, y in X])
    return X, Y


# ---------------------------------------------------------------- pendulum
def pendulum_energy(th, om):
    """H = omega^2 / 2 - cos(theta): -1 at rest, 1 on the separatrix."""
    return 0.5 * om ** 2 - np.cos(th)


def _rk4(f, s, dt):
    k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def pendulum_rhs(s):
    return np.array([s[1], -np.sin(s[0])])


def pendulum_trajectory(E, t_end, dt=0.005):
    """Start at the bottom moving right with energy E; returns (t, theta, omega)."""
    s = np.array([0.0, np.sqrt(2 * (E + 1))])
    ts, ss = [0.0], [s]
    for i in range(int(round(t_end / dt))):
        s = _rk4(pendulum_rhs, s, dt)
        ts.append((i + 1) * dt); ss.append(s)
    ss = np.array(ss)
    return np.array(ts), ss[:, 0], ss[:, 1]


def agm(a, b):
    while abs(a - b) > 1e-15:
        a, b = (a + b) / 2, np.sqrt(a * b)
    return a


def pendulum_period(E):
    """Exact period of libration: T = 4 K(k), k^2 = (1 + E) / 2, K via the AGM."""
    k = np.sqrt((1 + E) / 2)
    return 4 * np.pi / (2 * agm(1.0, np.sqrt(1 - k * k)))


def pendulum_period_numeric(E, dt=1e-3):
    """Four times the time from the bottom to the turning point (omega = 0)."""
    s, t = np.array([0.0, np.sqrt(2 * (E + 1))]), 0.0
    while True:
        n = _rk4(pendulum_rhs, s, dt)
        if n[1] <= 0:
            return 4 * (t + dt * s[1] / (s[1] - n[1]))
        s, t = n, t + dt


# ======================================================================= Part 2
def hv(D, rng):
    """A random bipolar hypervector in {-1, +1}^D."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


def cos(u, v):
    u, v = u.astype(float), v.astype(float)
    return float(u @ v / np.sqrt((u @ u) * (v @ v)))


def bind(*vs):
    """Elementwise product: multiplication by diag(v), orthogonal and self-inverse."""
    out = np.ones_like(vs[0])
    for v in vs:
        out = out * v
    return out


def bundle(*vs):
    """Majority vote, sign(sum). Odd count only, so there are no ties."""
    if len(vs) % 2 == 0:
        raise ValueError("bundle an odd number of vectors (or add a tie-breaker)")
    return np.sign(np.sum(np.array(vs, dtype=int), axis=0)).astype(np.int8)


def permute(v, k=1):
    """rho^k: cyclic shift, a permutation matrix (orthogonal, rho^D = I)."""
    return np.roll(v, k)


def flip(v, frac, rng):
    """Flip a fraction of the components (noise)."""
    out = v.copy()
    idx = rng.choice(len(v), size=int(round(frac * len(v))), replace=False)
    out[idx] *= -1
    return out


# ======================================================================= checks
def _check_all():
    lam, mu = 0.9, 0.5
    A, V = warmup_matrix(lam, mu)
    assert np.allclose(A, [[0.9, 0.4], [0, 0.5]]), A
    assert np.allclose(A @ V, V @ np.diag([lam, mu]))

    grid = np.linspace(-1, 1, 24)
    P = composition_matrix(grid, map1d)
    assert np.max(np.abs(P @ grid ** 2 - map1d(grid) ** 2)) < 5e-3

    rng = np.random.default_rng(0)
    for D in (64, 1024, 10000):
        s = cosine_samples(D, 20000, rng).std()
        assert abs(s * np.sqrt(D) - 1) < 0.03, (D, s)

    pts = np.random.default_rng(1).uniform(-2, 2, (100, 2))
    for ev, phi, _ in eigenfunctions(lam, mu):
        for x, y in pts:
            assert np.isclose(phi(*exact_map(x, y, lam, mu)), ev * phi(x, y))
    for x, y in pts:
        assert np.allclose(lift_matrix(lam, mu) @ lift(x, y), lift(*exact_map(x, y, lam, mu)))
    assert np.allclose(sorted(np.linalg.eigvals(koopman_coeff_matrix(lam, mu))),
                       sorted([lam, mu, lam ** 2]))

    X, Y = edmd_data(lam, mu, 300, np.random.default_rng(2))
    K3 = edmd(psi_exact, X, Y)
    assert np.allclose(K3, koopman_coeff_matrix(lam, mu), atol=1e-10)
    ev2 = sorted(np.linalg.eigvals(edmd(psi_dmd, X, Y)).real)
    assert abs(ev2[0] - mu) > 0.05, ev2             # DMD misses mu: the point of the beat

    for E in (-0.85, -0.4, 0.3, 0.85):
        assert np.isclose(pendulum_period(E), pendulum_period_numeric(E), rtol=1e-4)
    assert np.isclose(pendulum_period(-1 + 1e-12), 2 * np.pi)
    t, th, om = pendulum_trajectory(0.3, 10.0)
    assert np.ptp(pendulum_energy(th, om)) < 1e-6

    r = np.random.default_rng(3)
    D = 10000
    a, b, c, d = (hv(D, r) for _ in range(4))
    assert np.array_equal(bind(bind(a, b), b), a)
    assert abs(cos(bind(a, b), a)) < 0.04
    assert np.isclose(cos(bind(a, c), bind(b, c)), cos(a, b))
    s = bundle(a, b, c)
    assert all(abs(cos(s, v) - 0.5) < 0.04 for v in (a, b, c)) and abs(cos(s, d)) < 0.04
    assert abs(cos(flip(a, 0.3, r), a) - 0.4) < 1e-9
    assert abs(cos(permute(a), a)) < 0.04 and np.array_equal(permute(a, D), a)
    print("[models] all checks OK")
    print(f"  DMD eigenvalues {np.round(ev2, 4)}  vs Koopman {lam}, {mu}, {lam**2:.2f}")
    print("  pendulum periods", {E: round(pendulum_period(E), 3) for E in (-0.85, -0.4, 0.3, 0.85)})


if __name__ == "__main__":
    _check_all()
