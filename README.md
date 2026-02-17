# NLSBifurcation

Numerical bifurcation analysis and time dynamics for bound states of the one-dimensional nonlinear Schrodinger / Gross-Pitaevskii equation on the full line:

$$-\psi'' + V(x)\psi - |\psi|^2\psi = E\psi, \qquad x \in \mathbb{R}$$

where $V(x)$ is a compactly supported potential on $[a,b]$ and $E < 0$.

## Quick start

```bash
cd clean/
julia run.jl
```

Edit the parameter blocks at the top of `run.jl` to change the potential, continuation range, and dynamics settings.

## Pipeline

The workflow has three stages, each independently togglable in `run.jl`:

### Stage 1: Seed finding + continuation + plots

1. **Seed finding** -- Scan the Hamiltonian residual $H(\zeta)$ for sign changes at one or more trial energies $E_0$, yielding initial guesses for bound states.
2. **Branch continuation** -- Continue each seed in $E$ using BifurcationKit (PALC algorithm), producing full bifurcation branches.
3. **Plotting** -- Mass-energy curves $\mathcal{N}(E)$, $H^1$-norm curves, and spatial profiles $\psi_E(x)$.

### Stage 2: Spectral analysis (optional)

Compute eigenvalues of the linearized operators

$$L_+ = -\partial_x^2 + V(x) - E - 3\psi_0^2, \qquad L_- = -\partial_x^2 + V(x) - E - \psi_0^2$$

along each branch using finite differences on $[-X_{\max}, X_{\max}]$. Tracks the Vakhitov-Kolokolov stability criterion ($n(L_+) = 1$, $n(L_-) = 0$).

### Stage 3: Time dynamics (optional)

Evolve the time-dependent NLS

$$i\psi_t = -\psi'' + V(x)\psi - |\psi|^2\psi$$

using a symmetric split-step method with DST-I (Dirichlet BCs on $[-X_{\max}, X_{\max}]$).

Three initial conditions:

| IC type | Description |
|---|---|
| `:soliton` | Perturbed bound state: $(1+\varepsilon_1)\psi_0 + \varepsilon_2 h(x)$ |
| `:groundstate` | Ground eigenfunction of $-\partial_x^2$ on $[a,b]$, scaled to $\mathcal{N}[E_{\min}]$ |
| `:gaussian` | $A \exp(-x^2/2\sigma^2)$, scaled to $\mathcal{N}[E_{\min}]$ |

**Absorbing boundary conditions:** Setting `absorb_width > 0` adds a complex absorbing potential (CAP) near $\pm X_{\max}$. Outgoing radiation is damped rather than reflected off the hard walls. Controlled by `absorb_width`, `absorb_strength`, and `absorb_power`.

Output is an animated GIF saved to `results/dynamics/`.

## File structure

```
clean/
  NLSBifurcation.jl   Module definition, imports, includes, exports
  run.jl               Entry point with configurable parameter blocks
  globals.jl           Shared utilities (kappa, zeta parametrization, bisection)
  potentials.jl        Library of compactly supported potentials (15+ types)
  shooting.jl          ODE integration on [a,b] and Hamiltonian residual
  glue.jl              Tail matching and full-line solution assembly
  continuation.jl      Seed scanning and BifurcationKit continuation
  spectral.jl          L+/L- eigenvalue computation via finite differences
  dynamics.jl          Split-step time evolution with optional absorbing BCs
  plots.jl             All plotting routines (mass-energy, profiles, spectra)
  Project.toml         Julia package dependencies
```

## Dependencies

- [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl) -- ODE integration
- [BifurcationKit](https://github.com/bifurcationkit/BifurcationKit.jl) -- Numerical continuation
- [FFTW](https://github.com/JuliaMath/FFTW.jl) -- Discrete sine transform for split-step
- [Plots](https://github.com/JuliaPlots/Plots.jl) / LaTeXStrings / Colors -- Visualization

## Method overview

Bound states are homoclinic orbits of the ODE $u'' = (V(x) - E)u - u^3$. Outside the support $[a,b]$, solutions are sech-tails $\psi(x) = \pm A\,\mathrm{sech}(\kappa(x - x_0))$ with $A = \sqrt{-2E}$ and $\kappa = \sqrt{-E}$. The shooting method integrates through the potential region, then matches to tails by requiring the Hamiltonian residual $H = 0$ at $x = b$. The $\zeta$-parametrization $c = \sqrt{-2E}\tanh\zeta$ ensures the initial amplitude stays in the admissible range.

The split-step integrator uses the DST-I to diagonalize the kinetic operator $-\partial_x^2$ with Dirichlet BCs, giving an unconditionally stable scheme for the linear part. The symmetric splitting (half-step potential, full-step kinetic, half-step potential) is second-order accurate in $\Delta t$.
