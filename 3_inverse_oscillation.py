import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=32, depth=3):
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

    def forward(self, t):
        return self.net(t)


def grad(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def exact_solution(t, c=0.4, k=4.0, m=1.0, u0=1.0, v0=1.0):
    wn = np.sqrt(k / m)
    xi = c / (2.0 * m * wn)
    wd = wn * np.sqrt(1.0 - xi**2)
    return np.exp(-xi * wn * t) * (
        u0 * np.cos(wd * t)
        + (v0 + xi * wn * u0) / wd * np.sin(wd * t)
    )


def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    m = 1.0
    u0 = 1.0
    v0 = 1.0
    true_c = 0.4
    true_k = 4.0

    model = MLP().to(DEVICE)

    # Trainable positive physical parameters.
    log_c = nn.Parameter(torch.tensor(np.log(args.init_c), device=DEVICE))
    log_k = nn.Parameter(torch.tensor(np.log(args.init_k), device=DEVICE))

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [log_c, log_k],
        lr=args.lr
    )

    # Observations Γ_d, generated from the analytical solution.
    t_d_np = np.linspace(0, 15, args.n_d).reshape(-1, 1)
    u_d_np = exact_solution(t_d_np, true_c, true_k, m, u0, v0)
    Td = torch.tensor(t_d_np, device=DEVICE)
    Ud = torch.tensor(u_d_np, device=DEVICE)

    # Residual points Γ_f.
    tf_np = np.random.uniform(0, 15, size=(args.n_f, 1))
    Tf = torch.tensor(tf_np, device=DEVICE)

    # Initial-condition points Γ_b. Repeating t=0 is useful because this
    # second-order ODE has two initial constraints.
    Tb = torch.zeros((args.n_b, 1), device=DEVICE)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        c = torch.exp(log_c)
        k = torch.exp(log_k)

        # ODE residual.
        tf = Tf.clone().requires_grad_(True)
        u = model(tf)
        u_t = grad(u, tf)
        u_tt = grad(u_t, tf)
        residual = m * u_tt + c * u_t + k * u
        loss_f = torch.mean(residual**2)

        # Initial displacement and velocity.
        tb = Tb.clone().requires_grad_(True)
        ub = model(tb)
        ub_t = grad(ub, tb)
        loss_b = (
            torch.mean((ub - u0)**2)
            + torch.mean((ub_t - v0)**2)
        )

        # Observation loss.
        pred_d = model(Td)
        loss_d = torch.mean((pred_d - Ud)**2)

        # Same spirit as the paper: combine physics, BC/IC and data.
        loss = 0.25 * loss_f + 0.25 * loss_b + 0.5 * loss_d
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[P3] epoch={epoch:6d} "
                f"loss={loss.item():.4e} "
                f"ode={loss_f.item():.4e} "
                f"ic={loss_b.item():.4e} "
                f"data={loss_d.item():.4e} "
                f"c={c.item():.6f} "
                f"k={k.item():.6f}"
            )

    c_final = torch.exp(log_c).item()
    k_final = torch.exp(log_k).item()
    print("\n[P3] Parameter identification")
    print(f"True c = {true_c:.8f}, identified c = {c_final:.8f}")
    print(f"True k = {true_k:.8f}, identified k = {k_final:.8f}")

    # Plot prediction vs observations.
    t_plot = np.linspace(0, 15, 1000).reshape(-1, 1)
    with torch.no_grad():
        pred = model(torch.tensor(t_plot, device=DEVICE)).cpu().numpy().ravel()

    true = exact_solution(t_plot, true_c, true_k, m, u0, v0)

    plt.figure(figsize=(9, 5))
    plt.plot(t_plot.ravel(), true, label="True solution")
    plt.plot(t_plot.ravel(), pred, "--", label="PINN prediction")
    plt.scatter(t_d_np.ravel(), u_d_np.ravel(), s=18, label="Observations")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Inverse PINN — damped oscillator")
    plt.legend()
    plt.tight_layout()
    plt.savefig("problem3_result.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n-d", type=int, default=50)
    parser.add_argument("--n-f", type=int, default=100)
    parser.add_argument("--n-b", type=int, default=50)
    parser.add_argument("--init-c", type=float, default=0.8)
    parser.add_argument("--init-k", type=float, default=2.0)
    parser.add_argument("--print-every", type=int, default=2000)
    args = parser.parse_args(args=[])
    train(args)
