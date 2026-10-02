# PINNs vs Finite Differences for the Schrödinger Equation

Two scripts compare a physics-informed neural network (PINN) with a finite-difference solver. Both find the ground state of the quantum harmonic oscillator. Its exact answer is known, so every error is measured exactly.

| Script | Problem | Result |
|---|---|---|
| `schrodinger_pinn_vs_fd.py` | 1D oscillator | Finite differences win: L2 error 6.5e-4 in 0.8 ms (N=100). The PINN reaches 7.9e-4 in about 9 s. |
| `schrodinger_5d.py` | 5D oscillator | The PINN wins: L2 error 1.1e-3 in about 1 min, using < 0.1 GB. A 5D grid would need about 5 billion unknowns and 1.4 TB to match (extrapolated). |

Times are CPU time on a laptop and will vary with hardware.

## Run

```bash
pip install -r requirements.txt
python schrodinger_pinn_vs_fd.py   # a few seconds
python schrodinger_5d.py           # a few minutes
```

Each script prints its results and saves plots to a `figures/` folder.

## What the scripts do

- **Finite differences:** a second-order stencil on a grid, with SciPy's sparse eigensolver (`eigsh`). In 5D the grid has N⁵ points.
- **PINN:** a small `tanh` network trained with Adam, then L-BFGS. The energy comes from the Rayleigh quotient, and the boundary condition ψ = 0 is built into the network.
- **5D extras:** points are sampled from a Gaussian, and ψ = envelope × exp(NN) keeps the network out of excited states.

## Caveat

The 5D oscillator is separable. A classical method that uses this structure would beat both approaches. The PINN only beats a *generic* grid solver. This problem was chosen because its exact solution makes the errors measurable.
