"""
Problem 2 — Steady-state 2D heat conduction PINN

PDE:
    T_xx + T_yy + 1 = 0,  -1 < x < 1, -1 < y < 1
Boundary:
    T = 0 on the entire boundary.

The paper uses 1500 residual points, 500 boundary points,
4 hidden layers x 50 neurons, tanh, learning rate 0.0005,
Glorot uniform initialization, and the same general training setup
as Problem 1.

A convenient analytical solution for this square problem is NOT used
as the training target; the PINN is solving the PDE itself. The script
therefore reports PDE and boundary residuals. A finite-difference
reference is also generated for a visual error comparison.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=50, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._glorot_uniform)

    @staticmethod
    def _glorot_uniform(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


def grad(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def pde_residual(model, xy):
    xy.requires_grad_(True)
    T = model(xy)
    g = grad(T, xy)
    T_x = g[:, 0:1]
    T_y = g[:, 1:2]
    T_xx = grad(T_x, xy)[:, 0:1]
    T_yy = grad(T_y, xy)[:, 1:2]
    return T_xx + T_yy + 1.0


def make_fd_reference(n=101):
    """
    Solve -ΔT = 1 on [-1,1]^2 with T=0 on the boundary
    using a standard finite-difference linear system.
    """
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    h = x[1] - x[0]
    N = n - 2

    # Dense solve is simple but expensive for large n.
    # For n=81/101 it is still manageable for a mini-project.
    A = np.zeros((N * N, N * N))
    b = np.ones(N * N)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            q = idx(i, j)
            A[q, q] = 4.0
            if i > 0:
                A[q, idx(i - 1, j)] = -1.0
            if i < N - 1:
                A[q, idx(i + 1, j)] = -1.0
            if j > 0:
                A[q, idx(i, j - 1)] = -1.0
            if j < N - 1:
                A[q, idx(i, j + 1)] = -1.0

    interior = np.linalg.solve(A, b)
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = interior.reshape(N, N) * h**2
    X, Y = np.meshgrid(x, y)
    return X, Y, T


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Interior residual points.
    xy_f = np.random.uniform(-1, 1, size=(args.n_f, 2))
    Xf = torch.tensor(xy_f, device=DEVICE)

    # Random points on the four sides.
    n = args.n_b // 4
    s = np.random.uniform(-1, 1, size=(n, 1))
    left = np.hstack([-np.ones((n, 1)), s])
    right = np.hstack([np.ones((n, 1)), s])
    bottom = np.hstack([s, -np.ones((n, 1))])
    top = np.hstack([s, np.ones((n, 1))])
    xb = np.vstack([left, right, bottom, top])
    Xb = torch.tensor(xb, device=DEVICE)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()

        r = pde_residual(model, Xf.clone())
        loss_f = torch.mean(r**2)

        Tb = model(Xb)
        loss_b = torch.mean(Tb**2)

        loss = 0.5 * loss_f + 0.5 * loss_b
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[P2] epoch={epoch:6d} "
                f"total={loss.item():.4e} "
                f"pde={loss_f.item():.4e} "
                f"bc={loss_b.item():.4e}"
            )

    # Evaluate on a grid.
    n_test = args.grid
    x = np.linspace(-1, 1, n_test)
    y = np.linspace(-1, 1, n_test)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(
        np.column_stack([X.ravel(), Y.ravel()]), device=DEVICE
    )

    with torch.no_grad():
        pred = model(pts).cpu().numpy().reshape(X.shape)

    # FD reference.
    Xfd, Yfd, fd = make_fd_reference(args.fd_grid)
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (np.linspace(-1, 1, args.fd_grid), np.linspace(-1, 1, args.fd_grid)),
        fd,
        bounds_error=False,
        fill_value=None,
    )
    fd_on_test = interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    error = np.abs(pred - fd_on_test)

    print(f"[P2] mean absolute error vs FD = {error.mean():.6e}")
    print(f"[P2] max absolute error vs FD  = {error.max():.6e}")

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot_surface(Xfd, Yfd, fd, cmap="viridis")
    ax.set_title("Finite-difference reference")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("T")

    ax = fig.add_subplot(132, projection="3d")
    ax.plot_surface(X, Y, pred, cmap="viridis")
    ax.set_title("PINN solution")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("T")

    ax = fig.add_subplot(133)
    im = ax.imshow(
        error, origin="lower",
        extent=[-1, 1, -1, 1],
        aspect="equal", cmap="magma"
    )
    ax.set_title("|PINN - FD|")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("problem2_result.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n-f", type=int, default=1500)
    parser.add_argument("--n-b", type=int, default=500)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--fd-grid", type=int, default=81)
    parser.add_argument("--print-every", type=int, default=1000)
    args = parser.parse_args(args=[])
    train(args)
