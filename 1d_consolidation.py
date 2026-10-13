

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=50, depth=5):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._glorot)

    @staticmethod
    def _glorot(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xt):
        # Hard Dirichlet condition p(0,t)=0.
        x = xt[:, 0:1]
        return x * self.net(xt)


def grad(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def pde_residual(model, xt):
    xt.requires_grad_(True)
    p = model(xt)
    g = grad(p, xt)
    p_t = g[:, 1:2]
    p_x = g[:, 0:1]
    p_xx = grad(p_x, xt)[:, 0:1]
    return p_t - p_xx


def analytical_solution(x, t, terms=200):
    # p = sum over odd m: 4/(m*pi) sin(m*pi*x/2) exp(-m^2*pi^2*t/4)
    out = np.zeros_like(x, dtype=np.float64)
    for m in range(1, 2 * terms, 2):
        out += (
            4.0 / (m * np.pi)
            * np.sin(m * np.pi * x / 2.0)
            * np.exp(-(m * m) * np.pi**2 * t / 4.0)
        )
    return out


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collocation points Γ_f
    xf = np.random.rand(args.n_f, 1)
    tf = np.random.rand(args.n_f, 1) * args.t_max
    Xf = torch.tensor(np.hstack([xf, tf]), device=DEVICE)

    # Boundary x=1 for Neumann condition p_x(1,t)=0
    xb = np.ones((args.n_b, 1))
    tb = np.random.rand(args.n_b, 1) * args.t_max
    Xb = torch.tensor(np.hstack([xb, tb]), device=DEVICE)

    # Initial condition p(x,0)=1
    xi = np.random.rand(args.n_i, 1)
    ti = np.zeros((args.n_i, 1))
    Xi = torch.tensor(np.hstack([xi, ti]), device=DEVICE)

    history = []

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad()

        r = pde_residual(model, Xf.clone())
        loss_f = torch.mean(r**2)

        Xb_req = Xb.clone().requires_grad_(True)
        pb = model(Xb_req)
        pb_x = grad(pb, Xb_req)[:, 0:1]
        loss_b = torch.mean(pb_x**2)

        pi = model(Xi)
        loss_i = torch.mean((pi - 1.0)**2)

        # Paper reports wf=wb=1/4 and does not use data loss.
        # We additionally include the initial condition so the stated
        # mathematical problem is fully enforced.
        loss = 0.5 * loss_f + 0.25 * loss_b + 0.25 * loss_i
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % args.print_every == 0:
            history.append(float(loss.item()))
            print(
                f"[P1] epoch={epoch:6d} "
                f"total={loss.item():.4e} "
                f"pde={loss_f.item():.4e} "
                f"bc={loss_b.item():.4e} "
                f"ic={loss_i.item():.4e}"
            )

    # Optional L-BFGS refinement, corresponding to the paper's second stage.
    if args.lbfgs:
        print("[P1] Starting L-BFGS refinement...")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.lbfgs_iters,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            r = pde_residual(model, Xf.clone())
            lf = torch.mean(r**2)

            Xb_req = Xb.clone().requires_grad_(True)
            pb = model(Xb_req)
            pbx = grad(pb, Xb_req)[:, 0:1]
            lb = torch.mean(pbx**2)

            li = torch.mean((model(Xi) - 1.0)**2)
            loss = 0.5 * lf + 0.25 * lb + 0.25 * li
            loss.backward()
            return loss

        lbfgs.step(closure)

    # Test on a grid.
    nx, nt = 151, 101
    xx = np.linspace(0, 1, nx)
    tt = np.linspace(0, args.t_max, nt)
    X, T = np.meshgrid(xx, tt)
    pts = torch.tensor(
        np.column_stack([X.ravel(), T.ravel()]), device=DEVICE
    )
    with torch.no_grad():
        pred = model(pts).cpu().numpy().reshape(T.shape)

    exact = analytical_solution(X, T)
    error = np.abs(pred - exact)
    print(f"[P1] mean absolute error = {error.mean():.6e}")
    print(f"[P1] max absolute error  = {error.max():.6e}")

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot_surface(X, T, exact, cmap="viridis")
    ax.set_title("Analytical solution")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("p")

    ax = fig.add_subplot(132, projection="3d")
    ax.plot_surface(X, T, pred, cmap="viridis")
    ax.set_title("PINN solution")
    ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_zlabel("p")

    ax = fig.add_subplot(133)
    im = ax.imshow(
        error, origin="lower",
        extent=[0, 1, 0, args.t_max],
        aspect="auto", cmap="magma"
    )
    ax.set_title("Absolute error")
    ax.set_xlabel("x"); ax.set_ylabel("t")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("problem1_result.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--t-max", type=float, default=0.5)
    parser.add_argument("--n-f", type=int, default=500)
    parser.add_argument("--n-b", type=int, default=250)
    parser.add_argument("--n-i", type=int, default=125)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--lbfgs", action="store_true")
    parser.add_argument("--lbfgs-iters", type=int, default=1000)
    args = parser.parse_args(args=[])
    train(args)
