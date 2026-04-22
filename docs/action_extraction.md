# Action Extraction: Relative Pose → 6D Twist

RealEstate10K gives us a 3×4 camera extrinsic `E = [R | t]` per frame. For
supervised inverse-dynamics training we want a single 6D vector per
frame-to-frame step: the action `a = (v, ω) ∈ R^6`. This doc walks through
how `dataset.relative_pose` + `dataset.se3_log` produce that target.

## Step 1 — relative pose

For each consecutive pair, compose

```
E_rel = E_{t+1} · E_t^(-1)
```

using the analytic SE(3) inverse (not a general matrix inverse):

```
E_t^(-1) = [ R_t^T   -R_t^T t_t ]
           [  0             1   ]
```

`E_rel` is a 3×4 matrix (we drop the trailing `[0 0 0 1]` row). With
RealEstate10K's world-to-camera convention, `E_rel` maps a point expressed
in camera-`t` coordinates into camera-`(t+1)` coordinates.

Implemented in `dataset.relative_pose(P)`: builds `E` and `E_inv` in bulk,
multiplies `E[1:] @ E_inv[:-1]`, returns `(N-1, 3, 4)`.

## Step 2 — SE(3) log map

We then take the Lie algebra logarithm to collapse the 12 entries of `E_rel`
into 6 numbers. For any `T ∈ SE(3)` there's a unique twist
`ξ = (v, ω) ∈ R^6` (within a branch cut) such that `exp(ξ) = T`.

### Rotation part — axis-angle

```
θ = arccos( (tr(R) − 1) / 2 )            ∈ [0, π]
[ω]_× = (θ / (2 sin θ)) · (R − R^T)
ω = vee([ω]_×)                           ∈ R^3
```

`(tr(R) − 1)/2` is clamped to `[−1, 1]` before `arccos` because floating-
point rotation matrices drift slightly off SO(3).

### Translation part — inverse left Jacobian

The raw column `t` of `E_rel` is **not** the twist's `v`. They're related
by the left Jacobian of SO(3):

```
t = V · v
V = I + ((1 − cos θ)/θ²) [ω]_× + ((θ − sin θ)/θ³) [ω]_×²
```

So:

```
v = V^(-1) · t
V^(-1) = I − ½ [ω]_× + c₂ [ω]_×²
c₂ = (1 − (θ/2) cot(θ/2)) / θ²
```

### Small-angle handling

All of the coefficients above contain `sin θ / θ` or `(1 − cos θ)/θ²`-style
factors that are exactly singular at `θ = 0`. Between adjacent frames at
30 fps this is common (tiny pans, near-static shots). We switch to Taylor
expansions when `θ < 1e-4`:

| quantity                           | Taylor near 0              |
|------------------------------------|----------------------------|
| `θ / (2 sin θ)`                    | `1/2 + θ²/12 + …`          |
| `(1 − (θ/2) cot(θ/2)) / θ²`        | `1/12 + θ²/720 + …`        |

This keeps both branches numerically safe; `torch.where` selects per-element.

### Known edge case

At `θ = π` (180° rotation), `sin θ = 0` and the axis extraction is
genuinely singular — no choice of constants fixes it. Proper handling
requires recovering ω from the diagonal of `R` in that regime. We don't
handle it because consecutive-frame rotations on real video don't hit
180° in practice. If you ever see NaNs in `action`, check the θ histogram
for values near π.

## Shape summary

| name          | shape       | notes                              |
|---------------|-------------|------------------------------------|
| `P`           | `(N, 3, 4)` | per-frame extrinsics `[R | t]`     |
| `E_rel`       | `(N-1,3,4)` | from `relative_pose(P)`            |
| `a = (v, ω)`  | `(N-1, 6)`  | from `se3_log(E_rel)`              |

Layout inside `a`: first three entries are `v`, last three are `ω`.

## Velocity vs. twist

The twist is dimensionless relative to inter-frame spacing. If you want
actual linear/angular *velocity*, divide by Δt:

```python
dt = (timestamps[t+1] - timestamps[t]) * 1e-6   # seconds
velocity = action / dt
```

We store the twist (not velocity) because frame spacing is roughly
constant in RealEstate10K and the model's target scale is cleaner that
way. If you mix clips at different frame rates, switch to velocity.

## Why the twist, not the raw 3×4?

- **Minimal and invertible.** 6D ↔ `SE(3)` via `exp`/`log`, no redundant
  parameters and no constraints (vs. 9 rotation entries that have to be
  orthogonal).
- **Linear-ish in motion magnitude.** For small motions `ξ ≈ [t; axis·θ]`,
  which is approximately what the camera physically did. MSE in this
  space is well-behaved.
- **Regression-friendly.** The MLP head outputs 6 unbounded floats — no
  special handling for rotation matrices, quaternions, or SO(3)
  projection.

## Round-trip sanity

`exp(log(T)) = T` is a good unit check. A quick script:

```python
import torch
from dataset import se3_log, _skew

def se3_exp(xi):
    v, w = xi[..., :3], xi[..., 3:]
    theta = w.norm(dim=-1)
    small = theta < 1e-4
    W = _skew(w); W2 = W @ W
    I3 = torch.eye(3)
    a = torch.where(small, 1 - theta**2/6,    torch.sin(theta)/(theta+1e-20))
    b = torch.where(small, 0.5 - theta**2/24, (1-torch.cos(theta))/(theta**2+1e-20))
    c = torch.where(small, 1/6 - theta**2/120,(theta-torch.sin(theta))/(theta**3+1e-20))
    R = I3 + a[...,None,None]*W + b[...,None,None]*W2
    V = I3 + b[...,None,None]*W + c[...,None,None]*W2
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(*xi.shape[:-1], 3, 4); out[...,:3,:3]=R; out[...,:3,3]=t
    return out

xi = torch.randn(32, 6) * 0.3
err = (xi - se3_log(se3_exp(xi))).abs().max()
# err ~ 6e-8 in float32
```

This is the test that sits alongside `se3_log` when changing the
implementation.
