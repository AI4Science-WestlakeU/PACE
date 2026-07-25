# Repressilator ODE/SDE Trajectory Extension Design

## Goal

Append a self-contained teaching section to `Biocircuit101_notebook.ipynb` that compares deterministic and noisy three-dimensional repressilator trajectories from multiple initial conditions, explains what “steady state” means for an SDE, and numerically illustrates how noise smooths the deterministic Hopf transition.

## Scope

The existing notebook cells remain unchanged. New cells are appended after the current final section and reuse the notebook's NumPy, SciPy, Bokeh, and Plotly setup.

## Model and simulation

The deterministic drift remains

\[
\dot x=\frac{\beta}{1+z^n}-x,\qquad
\dot y=\frac{\beta}{1+x^n}-y,\qquad
\dot z=\frac{\beta}{1+y^n}-z.
\]

The stochastic model uses independent additive Itô noise,

\[
d\mathbf X=f(\mathbf X;\beta,n)\,dt+\sigma\,d\mathbf W,
\]

with reflection at zero to preserve non-negative concentrations. ODE trajectories use `odeint`; SDE trajectories use Euler–Maruyama with explicit seeds for reproducibility. ODE and SDE panels use the same initial conditions.

## Visual output

1. A side-by-side Plotly figure overlays multiple ODE and SDE trajectories in three-dimensional state space.
2. A Bokeh figure shows representative \(x(t),y(t),z(t)\) time series for direct temporal comparison.
3. A printed summary reports each path's initial condition, terminal state, post-burn-in amplitude, and spectral coherence.
4. A \((\beta,\sigma)\) numerical transition map reports post-burn-in amplitude and spectral coherence, with the deterministic Hopf threshold marked.

## Steady-state interpretation

Zeros of the drift remain deterministic landmarks after additive noise is introduced, but SDE sample paths do not converge to a point. If the deterministic Jacobian is Hurwitz and noise is small, the invariant distribution is concentrated near the fixed point. Its local covariance is approximated by the Ornstein–Uhlenbeck Lyapunov equation

\[
J\Sigma+\Sigma J^\top+\sigma^2 I=0.
\]

Above the deterministic Hopf boundary, probability concentrates around a noisy limit cycle. Below the boundary, noise may excite quasi-cycles. Stronger noise broadens the distribution and reduces phase coherence.

## Bifurcation interpretation

Because additive noise does not change the drift, the existing deterministic Hopf boundary remains the organizing skeleton. The notebook will not claim that an arbitrary numerical threshold is a rigorous stochastic bifurcation. It will distinguish:

- deterministic Hopf bifurcation;
- a P-bifurcation, defined through changes in stationary-density geometry;
- a D-bifurcation, defined through changes in random dynamical stability;
- the teaching-level observable transition map implemented here.

## Validation

Automated tests will extract only the appended core-code cell and verify:

- ODE and SDE output shapes;
- SDE reproducibility for a fixed seed;
- non-negative reflected SDE states;
- the analytical Hopf boundary;
- positive-semidefinite local OU covariance in the stable regime;
- finite trajectory summary statistics.

The notebook JSON will be parsed after editing, its appended core cell will be executed independently, and all appended plotting/analysis cells will be run with a non-interactive display configuration.
