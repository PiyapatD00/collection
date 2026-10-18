# Koopman operators and hyperdimensional computing: an animated explainer

A ~10-minute video (ManimGX) in three parts, for a viewer who knows linear algebra,
probability, calculus and a little algebraic geometry.

**Video:** [`video/koopman_hdc_explainer.mp4`](video/koopman_hdc_explainer.mp4) (1920x1080, 30 fps)

| Part | Scene | What it shows |
|---|---|---|
| 0 | `MathWarmup` | eigenvectors and diagonalization; iterating a linear map vs a nonlinear one; the composition operator `g -> g∘F` as a matrix on a grid; cosine of random ±1 vectors concentrating at `1/√D` |
| 1 | `KoopmanExactMap` | `F(x,y) = (λx, μy + (λ²−μ)x²)`; Koopman operator on observables; the invariant subspace `span{x, y, x²}`; level sets of `φ₂ = y − x²` permuted by F; eigenfunctions as coordinates straighten the dynamics |
| 1 | `KoopmanLift3D` | the lift `z = (x, y, x²)`: one step is a 3×3 matrix and the state stays on the variety `z₃ = z₁²` |
| 1 | `KoopmanEDMD` | Koopman from data: DMD with `(x, y)` misses μ (0.635 instead of 0.5) and its predictions drift; EDMD with `(x, y, x²)` is exact |
| 1 | `KoopmanPendulum` | energy is an eigenfunction with eigenvalue 1; the angle variable `e^{iα}` turns uniformly on each orbit; frequencies `Ω(E)` fill an arc of the unit circle (continuous spectrum) |
| 2 | `HDCBasics` | bipolar hypervectors, quasi-orthogonality and noise tolerance; binding (`diag(b)`, self-inverse); bundling (majority, cos ≈ ½); a role–filler record queried through a cleanup memory; permutation for order; bridge back to Koopman |

## Every number is checked before a frame is drawn

`models.py` holds all the math (pure numpy). Each scene's `verify()` re-checks what it
will show and refuses to render if a claim is false, for example:

- `φ(F(p)) = λ φ(p)` for the three eigenfunctions, on random points
- `A · lift(p) = lift(F(p))`, and lifted orbits stay on `z₃ = z₁²`
- EDMD with `(x, y, x²)` recovers `{0.5, 0.81, 0.9}` to 1e-9; DMD does not
- pendulum period `4K(k)` (via the AGM) agrees with an RK4 integration to 1e-4
- at D = 10,000: unbinding is exact, bundle similarities ≈ 0.5, cleanup returns the
  right filler, `cos(abc, cba) ≈ 0`

```bash
python3 models.py          # all checks, no rendering
python3 part1_koopman.py   # one chapter's verify()
```

## Rendering

Needs Python ≥ 3.13 and `pip install manimgx` (no LaTeX, display or GPU needed).

```bash
bash make_video.sh                  # all chapters at 1920x1080/30 fps, joined into video/
bash make_video.sh 640x360 30       # quick draft
python3 render_check.py part2_hdc.py HDCBasics --at 40 90    # stills of one scene
```

On a CPU-only machine (4 cores, no GPU) the final render takes about 15 minutes;
drafts take a few.

## Files

| File | Role |
|---|---|
| `models.py` | the math: maps, Koopman matrices, EDMD, pendulum, hypervector algebra, checks |
| `pattern_base.py` | scene skeleton from the manim-animation-pattern skill: beats, `verify()`, pinned captions |
| `lesson_base.py` | shared look: palette (one meaning per color), captions, headers, chapter cards |
| `part0_math.py`, `part1_koopman.py`, `part2_hdc.py` | one lesson class per scene; each `beat_*` method is one storyboard step |
| `render_check.py` | render + contact sheet + ManimGX layout report |
| `make_video.sh` | renders everything and joins the chapters |

To change a beat, override it in a subclass or edit the one `beat_*` method; to change
the example (λ, μ, D, the record's roles), edit the scene's `DEFAULTS`/`PROBLEM`.
