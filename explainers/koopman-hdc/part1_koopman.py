"""
Part 1 -- the Koopman operator.

    KoopmanExactMap   a nonlinear map with exact polynomial eigenfunctions (2D)
    KoopmanLift3D     the same map lifted to z = (x, y, x^2): a linear map of R^3 (3D)
    KoopmanEDMD       finding K from data; the dictionary decides (2D)
    KoopmanPendulum   energy and angle eigenfunctions, a continuous spectrum (2D)

    python3 part1_koopman.py                                          # math check only
    python3 render_check.py part1_koopman.py KoopmanExactMap --draft  # one scene
"""
from lesson_base import *
import models as M

MAP = dict(lam=0.9, mu=0.5)
MAP_TEX = r"F(x,y)=\big(\lambda x,\ \mu y+(\lambda^2-\mu)\,x^2\big)"


def scene_map(ax, f):
    """Lift a map of data coordinates to scene points (for ApplyPointwiseFunction)."""
    return lambda q: ax.c2p(*f(*ax.p2c(q)[:2]))


def polyline(ax, pts, **style):
    return VMobject(**style).set_points_as_corners([ax.c2p(*q) for q in pts])


# =========================================================================== exact map
class ExactMapLesson(Lesson):
    DEFAULTS = dict(
        **MAP,
        starts=[(-2.0, 3.6), (2.0, 2.8), (-1.6, 1.4), (1.4, 4.2),
                (0.7, -0.9), (-0.6, 3.8), (1.9, 4.3), (-1.0, -0.4)],
        n_iter=20,
        levels=[-1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0],
        table_start=0, table_rows=4,
        box=((-2.2, 2.2), (-1.5, 4.5)),
    )
    BEATS = ["title", "phase_portrait", "definition", "invariant_subspace",
             "level_sets", "eigen_coords"]
    PANEL_LEFT = 0.4

    # ---------------------------------------------------------- pure math
    def setup_model(self):
        p = self.p
        lam, mu = p["lam"], p["mu"]
        self.F = lambda x, y: M.exact_map(x, y, lam, mu)
        self.Phi = lambda x, y: (x, y - x ** 2)
        self.orbits = [M.orbit(*s, lam, mu, p["n_iter"]) for s in p["starts"]]
        self.Kc = M.koopman_coeff_matrix(lam, mu)
        self.efs = M.eigenfunctions(lam, mu)

    def verify(self):
        p = self.p
        (xa, xb), (ya, yb) = p["box"]
        for o in self.orbits:
            if not (np.all((o[:, 0] >= xa) & (o[:, 0] <= xb) & (o[:, 1] >= ya) & (o[:, 1] <= yb))):
                raise ValueError(f"orbit from {o[0]} leaves the plotted box")
            if not np.all((o[:, 1] - o[:, 0] ** 2 >= ya) & (o[:, 1] - o[:, 0] ** 2 <= yb)):
                raise ValueError(f"orbit from {o[0]} leaves the box in eigen-coordinates")
        rng = np.random.default_rng(5)
        for ev, phi, tex in self.efs:
            for x, y in rng.uniform(-2, 2, (50, 2)):
                if not np.isclose(phi(*self.F(x, y)), ev * phi(x, y)):
                    raise ValueError(f"{tex} is not an eigenfunction with eigenvalue {ev}")
        evs = sorted(np.linalg.eigvals(self.Kc).real)
        if not np.allclose(evs, sorted([p["lam"], p["mu"], p["lam"] ** 2])):
            raise ValueError(f"eigenvalues {evs}")
        print(f"[verify] eigenvalues {np.round(evs, 4)}; phi(F(p)) = ev phi(p) on 50 points")

    # ---------------------------------------------------------- factories
    def make_axes(self):
        (xa, xb), (ya, yb) = self.p["box"]
        ax = Axes(x_range=(xa, xb, 1), y_range=(ya, yb, 1), x_length=5.8, y_length=5.8,
                  axis_config=self.AXIS_CONFIG).shift(3.5 * LEFT + 0.25 * DOWN)
        return ax

    def axis_labels(self, xl, yl):
        ax = self.ax
        a = MathTex(xl, font_size=34).next_to(ax.x_axis.get_end(), DOWN, buff=0.15)
        b = MathTex(yl, font_size=34).next_to(ax.y_axis.get_end(), RIGHT, buff=0.15)
        return VGroup(a, b)

    def level_curve(self, c, width=None):
        ax, (xa, xb) = self.ax, self.p["box"][0]
        xm = min(xb, np.sqrt(self.p["box"][1][1] - c))
        col = C_EIG1 if c == 0 else interpolate_color(ManimColor(C_EIG2), ManimColor(C_NEG),
                                                      (c + 1.2) / 4.2)
        return ax.plot(lambda x: x ** 2 + c, x_range=[-xm, xm], color=col,
                       stroke_width=width or (5 if c == 0 else 2.5))

    def panel(self, texs, font_size=34, top=2.9, max_w=6.2, buff=0.3):
        col = VGroup(*[MathTex(t, font_size=font_size) for t in texs])
        col.arrange(DOWN, aligned_edge=LEFT, buff=buff)
        if col.width > max_w:
            col.scale_to_fit_width(max_w)
        col.move_to(3.6 * RIGHT).align_to(top * UP, UP).align_to(self.PANEL_LEFT * RIGHT, LEFT)
        return col

    def fit_right(self, mob, left=None):
        """Scale mob so it fits between `left` and the frame's right margin."""
        left = self.PANEL_LEFT if left is None else left
        room = config.frame_width / 2 - 0.35 - left
        if mob.width > room:
            mob.scale_to_fit_width(room)
        return mob.align_to(left * RIGHT, LEFT)

    # ---------------------------------------------------------- beats
    def beat_title(self):
        self.chapter("Part 1", "The Koopman operator",
                     r"linear algebra for nonlinear dynamics")

    def beat_phase_portrait(self):
        p = self.p
        self.header("A nonlinear map of the plane")
        self.ax = self.make_axes()
        self.labels = self.axis_labels("x", "y")
        f = self.panel([MAP_TEX, rf"\lambda={p['lam']:g},\quad \mu={p['mu']:g}"], font_size=34)
        self.play(Create(self.ax), FadeIn(self.labels), Write(f[0]))
        self.play(FadeIn(f[1]))
        self.Fpanel = f
        dots = VGroup(*[Dot(self.ax.c2p(*o[0]), radius=0.06, color=WHITE) for o in self.orbits])
        self.play(FadeIn(dots, lag_ratio=0.1))
        trails = VGroup()
        for n in range(p["n_iter"]):
            segs = [Line(self.ax.c2p(*o[n]), self.ax.c2p(*o[n + 1]), color=GREY_B, stroke_width=2)
                    for o in self.orbits]
            trails.add(*segs)
            self.play(*[d.animate.move_to(self.ax.c2p(*o[n + 1])) for d, o in zip(dots, self.orbits)],
                      *[Create(s) for s in segs], run_time=0.5 if n < 3 else 0.22)
        self.say(r"Every orbit is pulled onto the parabola $y=x^2$, then slides to the origin.")
        self.say(r"Is there a change of coordinates in which this map becomes \emph{linear}?")
        self.play(FadeOut(trails), FadeOut(dots))

    def beat_definition(self):
        p = self.p
        self.header("The Koopman operator acts on observables")
        defs = self.panel([
            r"\text{observable: } g:\mathbb{R}^2\to\mathbb{R}",
            r"(\mathcal{K}g)(p)=g\big(F(p)\big)",
            r"\mathcal{K}(\alpha g+\beta h)=\alpha\,\mathcal{K}g+\beta\,\mathcal{K}h",
            r"\mathcal{K}\varphi=\lambda\varphi\ \Longleftrightarrow\ \varphi\big(F(p)\big)=\lambda\,\varphi(p)",
        ], font_size=32, top=1.6)
        defs[3].set_color(C_EIG1)
        self.play(self.Fpanel.animate.scale(0.8).to_edge(UP, buff=0.9).align_to(defs, LEFT))
        for m in defs[:3]:
            self.play(Write(m), run_time=0.9)
        self.say(r"Instead of moving points, $\mathcal{K}$ moves measurements: "
                 r"$g$ evaluated one step later. It is linear in $g$.")
        self.play(Write(defs[3]))
        self.say(r"An \emph{eigenfunction} is a measurement that gets multiplied by the same "
                 r"$\lambda$ at every step, from every starting point.")

        # a concrete check: phi_2 = y - x^2 halves along an orbit while x, y look messy
        o = self.orbits[p["table_start"]]
        k = p["table_rows"]
        pts = VGroup(*[Dot(self.ax.c2p(*q), radius=0.07, color=C_EIG2) for q in o[:k]])
        arrows = VGroup(*[Arrow(self.ax.c2p(*o[i]), self.ax.c2p(*o[i + 1]), buff=0.08,
                                color=C_EIG2, stroke_width=3, max_tip_length_to_length_ratio=0.2)
                          for i in range(k - 1)])
        rows = [r"n", r"x_n", r"y_n", r"y_n-x_n^2"]
        cols = [[rf"{i}", rf"{o[i, 0]:.3f}", rf"{o[i, 1]:.3f}", rf"{o[i, 1] - o[i, 0] ** 2:.3f}"]
                for i in range(k)]
        tab = VGroup()
        for r, name in enumerate(rows):
            line = VGroup(MathTex(name, font_size=34), *[MathTex(c[r], font_size=32) for c in cols])
            tab.add(line)
        grid = VGroup(*[m for line in tab for m in line])
        grid.arrange_in_grid(rows=len(rows), cols=k + 1, buff=(0.3, 0.18),
                             col_alignments="l" + "r" * k)
        tab[3].set_color(C_EIG2)
        self.fit_right(grid.next_to(defs, DOWN, buff=0.45))
        self.play(FadeIn(pts[0]))
        for i in range(k - 1):
            self.play(GrowArrow(arrows[i]), FadeIn(pts[i + 1]), run_time=0.6)
        self.play(FadeIn(grid, lag_ratio=0.02), run_time=1.2)
        self.say(r"Along this orbit $x_n$, $y_n$ change in a tangled way, but "
                 r"$\varphi_2=y-x^2$ simply halves: $\mu=0.5$.", wait=4)
        self.play(FadeOut(VGroup(pts, arrows, grid, defs)))

    def beat_invariant_subspace(self):
        self.header(r"A finite piece of $\mathcal{K}$: three observables")
        lines = self.panel([
            r"\mathcal{K}\,x=\lambda\,x",
            r"\mathcal{K}\,y=\mu\,y+(\lambda^2-\mu)\,x^2",
            r"\mathcal{K}\,x^2=\lambda^2\,x^2",
            r"\begin{pmatrix}c_x\\c_y\\c_{x^2}\end{pmatrix}\mapsto"
            r"\begin{pmatrix}\lambda&0&0\\0&\mu&0\\0&\lambda^2-\mu&\lambda^2\end{pmatrix}"
            r"\begin{pmatrix}c_x\\c_y\\c_{x^2}\end{pmatrix}",
        ], font_size=32, top=1.9)
        for m in lines[:3]:
            self.play(Write(m), run_time=0.9)
        self.say(r"Apply $\mathcal{K}$ to $x$, $y$, $x^2$: nothing new appears. "
                 r"$\mathrm{span}\{x,y,x^2\}$ is invariant.")
        self.play(Write(lines[3]), run_time=1.5)
        self.say(r"On $g=c_x x+c_y y+c_{x^2}x^2$, $\mathcal{K}$ is a $3\times3$ matrix. "
                 r"Its eigenvectors are eigenfunctions.")
        ev = self.panel([
            r"\lambda=0.9:\quad \varphi_1=x",
            r"\mu=0.5:\quad \varphi_2=y-x^2",
            r"\lambda^2=0.81:\quad \varphi_3=x^2",
        ], font_size=32, top=1.9)
        for m, c in zip(ev, [C_EIG1, C_EIG2, C_EIG3]):
            m.set_color(c)
        self.play(FadeOut(lines[:3]), lines[3].animate.next_to(ev, DOWN, buff=0.5).align_to(ev, LEFT),
                  FadeIn(ev, lag_ratio=0.3), run_time=1.5)
        check = MathTex(r"\mathcal{K}(y-x^2)=\mu y+(\lambda^2-\mu)x^2-\lambda^2x^2=\mu\,(y-x^2)",
                        font_size=28).next_to(lines[3], DOWN, buff=0.45)
        self.fit_right(check)
        self.play(Write(check))
        self.say(r"Check: $\mathcal{K}(y-x^2)=\mu(y-x^2)$. Three eigenvalues, three exact "
                 r"eigenfunctions.")
        self.play(FadeOut(lines[3]), FadeOut(check))
        self.ev_panel = ev

    def beat_level_sets(self):
        p = self.p
        self.header(r"Level sets of $\varphi_2=y-x^2$")
        fam = VGroup(*[self.level_curve(c) for c in p["levels"]])
        tags = VGroup(*[MathTex(rf"{c:g}", font_size=24).set_color(cv.get_color())
                        .next_to(cv.get_end(), UP, buff=0.05)
                        for c, cv in zip(p["levels"], fam) if c in (0.0, 1.2, 3.0)])
        self.play(Create(fam, lag_ratio=0.1), FadeIn(tags), run_time=2)
        note = self.panel([r"\varphi_2=c\ \xrightarrow{\ F\ }\ \varphi_2=\mu c"],
                          font_size=34, top=-0.4)
        note.set_color(C_EIG2)
        self.say(r"Curves $y=x^2+c$: the sets where $\varphi_2$ is constant.", wait=2)
        Fm = scene_map(self.ax, self.F)
        self.play(FadeOut(tags), Write(note))
        for _ in range(2):
            self.play(ApplyPointwiseFunction(Fm, fam), run_time=2)
            self.wait(0.5)
        self.say(r"$F$ maps each level set onto another one with $c$ halved. "
                 r"$\varphi_2=0$, the parabola, is invariant.")
        self.say(r"Level sets of polynomial eigenfunctions are algebraic curves, and "
                 r"the dynamics permutes them.", wait=3)
        self.play(FadeOut(fam), FadeOut(note))

    def beat_eigen_coords(self):
        p = self.p
        self.header(r"Eigenfunctions as coordinates")
        fam = VGroup(*[self.level_curve(c) for c in p["levels"]])
        lines = VGroup(*[polyline(self.ax, o, stroke_color=GREY_B, stroke_width=1.5)
                         for o in self.orbits])
        dots = VGroup(*[Dot(self.ax.c2p(*q), radius=0.035, color=WHITE)
                        for o in self.orbits for q in o])
        self.play(FadeIn(fam), FadeIn(lines), FadeIn(dots, lag_ratio=0.002), run_time=1.5)
        self.say(r"Orbits and level sets together. Now use $(\varphi_1,\varphi_2)=(x,\ y-x^2)$ "
                 r"as coordinates.", wait=2.5)
        Pm = scene_map(self.ax, self.Phi)
        new_labels = self.axis_labels(r"\varphi_1", r"\varphi_2")
        new_labels[0].set_color(C_EIG1); new_labels[1].set_color(C_EIG2)
        self.play(ApplyPointwiseFunction(Pm, VGroup(fam, lines)),
                  *[d.animate.move_to(Pm(d.get_center())) for d in dots],
                  Transform(self.labels, new_labels), run_time=3)
        diag = self.panel([r"\varphi_1\mapsto\lambda\,\varphi_1,\quad\varphi_2\mapsto\mu\,\varphi_2",
                           r"\begin{pmatrix}\varphi_1\\\varphi_2\end{pmatrix}\mapsto"
                           r"\begin{pmatrix}0.9&0\\0&0.5\end{pmatrix}"
                           r"\begin{pmatrix}\varphi_1\\\varphi_2\end{pmatrix}"],
                          font_size=34, top=-0.2)
        self.play(Write(diag))
        self.say(r"The parabolas straighten and the map becomes a diagonal matrix: "
                 r"the Part 0 picture, recovered.")
        self.say(r"Here they are global coordinates. In general they are local or approximate, "
                 r"so we compute them.", wait=3.5)
        self.clear_stage()
        self.drop_all_text()


class KoopmanExactMap(ExactMapLesson):
    PROBLEM = dict()


# =========================================================================== 3D lift
class LiftLesson(ThreeDLesson):
    DEFAULTS = dict(**MAP, starts=[(-2.0, 3.6), (2.0, 2.8), (-1.6, 1.4), (1.4, 4.2)],
                    n_iter=16, box=((-2.2, 2.2), (-1.5, 4.5), (0, 4.8)))
    BEATS = ["lift", "linear_orbit"]

    def setup_model(self):
        p = self.p
        self.A = M.lift_matrix(p["lam"], p["mu"])
        self.orbits = [M.orbit(*s, p["lam"], p["mu"], p["n_iter"]) for s in p["starts"]]
        self.zorbits = []
        for s in p["starts"]:
            z = [M.lift(*s)]
            for _ in range(p["n_iter"]):
                z.append(self.A @ z[-1])
            self.zorbits.append(np.array(z))

    def verify(self):
        for o, z in zip(self.orbits, self.zorbits):
            if not np.allclose(np.array([M.lift(*q) for q in o]), z):
                raise ValueError("A^n lift(p) != lift(F^n(p))")
            if not np.allclose(z[:, 2], z[:, 0] ** 2):
                raise ValueError("lifted orbit leaves the surface z3 = z1^2")
        ev = sorted(np.linalg.eigvals(self.A).real)
        print(f"[verify] lifted orbits stay on z3 = z1^2; eigenvalues of A {np.round(ev, 4)}")

    def beat_lift(self):
        (xa, xb), (ya, yb), (za, zb) = self.p["box"]
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=0.62)
        ax = self.ax = ThreeDAxes(x_range=(xa, xb, 1), y_range=(ya, yb, 1), z_range=(za, zb, 1),
                                  x_length=4.4, y_length=6, z_length=4.8,
                                  axis_config=self.AXIS_CONFIG)
        ax.shift(-ax.c2p(0, 1.5, 1.6))
        self.header(r"Lift the state: $z=(x,\ y,\ x^2)\in\mathbb{R}^3$")
        self.ocols = [C_POS, C_EIG2, C_EIG1, C_GOOD]
        dots = VGroup(*[Dot3D(ax.c2p(*o[0], 0), radius=0.08, color=c)
                        for o, c in zip(self.orbits, self.ocols)])
        flat = VGroup(*[VMobject(stroke_color=c, stroke_width=2.5)
                        .set_points_as_corners([ax.c2p(*q, 0) for q in o])
                        for o, c in zip(self.orbits, self.ocols)])
        self.play(Create(ax), FadeIn(dots), Create(flat), run_time=1.5)
        self.say(r"The orbits from before, in the $(x,y)$ plane. Add a third coordinate "
                 r"$z_3=x^2$.", wait=2)
        self.move_camera(phi=72 * DEGREES, theta=-112 * DEGREES, run_time=2.5)
        surf = surface(lambda u, v: ax.c2p(u, v, u * u), u_range=(-2.15, 2.15),
                       v_range=(ya, yb), resolution=res(24, 12), color=C_EIG3, opacity=0.16)
        self.play(FadeIn(surf),
                  *[d.animate.move_to(ax.c2p(*z[0])) for d, z in zip(dots, self.zorbits)],
                  run_time=2)
        self.play(FadeOut(flat), run_time=0.6)
        self.surf, self.dots = surf, dots

    def beat_linear_orbit(self):
        ax = self.ax
        A = self.hud(MathTex(r"z_{n+1}=\begin{pmatrix}0.9&0&0\\0&0.5&0.31\\0&0&0.81\end{pmatrix}z_n",
                             font_size=34).to_corner(UR, buff=0.4).shift(0.5 * DOWN))
        self.play(FadeIn(A))
        self.say(r"In $\mathbb{R}^3$ one step is a \emph{matrix}: "
                 r"$z_{n+1}=Az_n$, exactly, for every starting point.", wait=2)
        trails = VGroup()
        for n in range(self.p["n_iter"]):
            segs = [Line(ax.c2p(*z[n]), ax.c2p(*z[n + 1]), color=c, stroke_width=3)
                    for z, c in zip(self.zorbits, self.ocols)]
            trails.add(*segs)
            self.play(*[d.animate.move_to(ax.c2p(*z[n + 1])) for d, z in zip(self.dots, self.zorbits)],
                      *[Create(s) for s in segs], run_time=0.5 if n < 3 else 0.2)
        self.begin_ambient_camera_rotation(rate=0.15)
        self.say(r"The state never leaves the surface $z_3=z_1^2$: an algebraic variety "
                 r"that $A$ maps into itself.")
        self.say(r"Eigenvalues of $A$: $0.9,\ 0.5,\ 0.81$, the Koopman eigenvalues. "
                 r"Nonlinear in $\mathbb{R}^2$, linear in $\mathbb{R}^3$.", wait=3)
        self.stop_ambient_camera_rotation()
        self.drop_all_text()
        self.play(FadeOut(VGroup(ax, self.surf, self.dots, trails, A)))


class KoopmanLift3D(LiftLesson):
    PROBLEM = dict()


# =========================================================================== EDMD
class EDMDLesson(Lesson):
    DEFAULTS = dict(**MAP, n_data=300, n_show=60, seed=2, p0=(-2.0, 3.6), n_pred=12,
                    box=((-2.2, 2.2), (-1.5, 4.5)))
    BEATS = ["data", "fit", "spectrum", "predict"]

    def setup_model(self):
        p = self.p
        lam, mu = p["lam"], p["mu"]
        self.X, self.Y = M.edmd_data(lam, mu, p["n_data"], np.random.default_rng(p["seed"]))
        self.K2 = M.edmd(M.psi_dmd, self.X, self.Y)
        self.K3 = M.edmd(M.psi_exact, self.X, self.Y)
        self.ev2 = np.sort(np.linalg.eigvals(self.K2).real)
        self.ev3 = np.sort(np.linalg.eigvals(self.K3).real)
        self.true_ev = np.sort([lam, mu, lam ** 2])
        self.truth = M.orbit(*p["p0"], lam, mu, p["n_pred"])
        self.pred2 = M.edmd_predict(M.psi_dmd, self.K2, p["p0"], p["n_pred"])
        self.pred3 = M.edmd_predict(M.psi_exact, self.K3, p["p0"], p["n_pred"])

    def verify(self):
        if not np.allclose(self.ev3, self.true_ev, atol=1e-9):
            raise ValueError(f"EDMD eigenvalues {self.ev3} != {self.true_ev}")
        if not np.allclose(self.pred3, self.truth, atol=1e-9):
            raise ValueError("EDMD prediction is not exact")
        err2 = np.abs(self.pred2 - self.truth).max()
        if err2 < 0.3:
            raise ValueError(f"DMD error only {err2:.2f}: the contrast would not show")
        print(f"[verify] EDMD eigenvalues {np.round(self.ev3, 4)} exact; DMD {np.round(self.ev2, 4)}, "
              f"max prediction error {err2:.2f}")

    def make_axes(self):
        (xa, xb), (ya, yb) = self.p["box"]
        return Axes(x_range=(xa, xb, 1), y_range=(ya, yb, 1), x_length=5.4, y_length=5.4,
                    axis_config=self.AXIS_CONFIG).shift(3.7 * LEFT + 0.3 * DOWN)

    def mat_tex(self, K, name):
        rows = r"\\".join("&".join(f"{v:.3f}".replace("-0.000", "0.000") for v in r) for r in K)
        return rf"{name}=\begin{{pmatrix}}{rows}\end{{pmatrix}}"

    def beat_data(self):
        p = self.p
        self.header("Koopman from data: EDMD")
        ax = self.ax = self.make_axes()
        lab = VGroup(MathTex("x", font_size=32).next_to(ax.x_axis.get_end(), DOWN, 0.15),
                     MathTex("y", font_size=32).next_to(ax.y_axis.get_end(), RIGHT, 0.15))
        arrows = VGroup(*[Arrow(ax.c2p(*a), ax.c2p(*b), buff=0, stroke_width=2, color=C_DATA,
                                tip_length=0.1, max_tip_length_to_length_ratio=0.3,
                                max_stroke_width_to_length_ratio=8)
                          for a, b in zip(self.X[:p["n_show"]], self.Y[:p["n_show"]])])
        starts = VGroup(*[Dot(ax.c2p(*a), radius=0.035, color=WHITE) for a in self.X[:p["n_show"]]])
        arrows = VGroup(arrows, starts)
        self.play(Create(ax), FadeIn(lab))
        self.play(FadeIn(arrows, lag_ratio=0.01), run_time=2)
        self.say(r"Suppose we only have data: snapshot pairs $p_i\to F(p_i)$, and no formula "
                 rf"({p['n_data']} pairs, {p['n_show']} drawn).")
        self.data_mobs = VGroup(ax, lab, arrows)

    def beat_fit(self):
        rows = VGroup(
            MathTex(r"\Psi(p)=\big(\psi_1(p),\dots,\psi_k(p)\big)\ \ \text{(dictionary)}", font_size=30),
            MathTex(r"K=\arg\min_K\sum_i\big\|\Psi(F(p_i))-\Psi(p_i)\,K\big\|^2", font_size=30),
            MathTex(r"\text{DMD:}\ \ \Psi=(x,\,y)", font_size=30).set_color(C_BAD),
            MathTex(self.mat_tex(self.K2, r"K_2"), font_size=30).set_color(C_BAD),
            MathTex(r"\text{EDMD:}\ \ \Psi=(x,\,y,\,x^2)", font_size=30).set_color(C_GOOD),
            MathTex(self.mat_tex(self.K3, r"K_3"), font_size=30).set_color(C_GOOD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        if rows.width > 6.6:
            rows.scale_to_fit_width(6.6)
        rows.move_to(3.4 * RIGHT + 0.2 * UP)
        self.play(Write(rows[0]))
        self.play(Write(rows[1]))
        self.say(r"Choose a dictionary of observables, then solve one linear least-squares "
                 r"problem for $K$.")
        self.play(FadeIn(rows[2]), FadeIn(rows[3]))
        self.play(FadeIn(rows[4]), FadeIn(rows[5]))
        self.say(r"Two dictionaries: plain $(x,y)$ (this is DMD), and one that adds $x^2$.")
        self.fit_rows = rows

    def beat_spectrum(self):
        self.play(FadeOut(self.data_mobs), self.fit_rows.animate.scale(0.85).to_edge(RIGHT, buff=0.3)
                  .to_edge(UP, buff=0.9))
        nl = NumberLine(x_range=(0.4, 1.0, 0.1), length=6.2, include_numbers=True,
                        decimal_number_config=dict(num_decimal_places=1), font_size=26,
                        color=GREY_B).shift(3.4 * LEFT + 0.6 * DOWN)
        title = Tex("eigenvalues", font_size=30).next_to(nl, UP, buff=1.6)
        true = VGroup(*[Circle(0.14, color=WHITE, stroke_width=3).move_to(nl.n2p(v))
                        for v in self.true_ev])
        dmd = VGroup(*[Cross(Square(0.22), stroke_color=C_BAD, stroke_width=4)
                       .move_to(nl.n2p(v) + 0.75 * UP) for v in self.ev2])
        edmd = VGroup(*[Dot(nl.n2p(v), radius=0.08, color=C_GOOD) for v in self.ev3])
        legend = VGroup(
            VGroup(Circle(0.1, color=WHITE, stroke_width=3), Tex("Koopman (exact)", font_size=26)).arrange(RIGHT, buff=0.2),
            VGroup(Cross(Square(0.16), stroke_color=C_BAD, stroke_width=4), Tex("DMD", font_size=26)).arrange(RIGHT, buff=0.2),
            VGroup(Dot(radius=0.08, color=C_GOOD), Tex("EDMD", font_size=26)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(nl, DOWN, buff=0.7)
        tags = VGroup(*[MathTex(t, font_size=28).next_to(c, DOWN, buff=0.45)
                        for t, c in zip([r"\mu", r"\lambda^2", r"\lambda"], true)])
        self.play(Create(nl), FadeIn(title), FadeIn(true), FadeIn(tags), FadeIn(legend[0]))
        self.play(FadeIn(dmd), FadeIn(legend[1]))
        self.say(rf"DMD finds $0.9$ but puts the second eigenvalue at {self.ev2[0]:.3f}, not "
                 r"$0.5$: $(x,y)$ is not invariant under $\mathcal{K}$.")
        self.play(FadeIn(edmd, scale=2), FadeIn(legend[2]))
        self.say(r"With $x^2$ included the subspace is invariant, and EDMD recovers "
                 r"$0.5,\ 0.81,\ 0.9$ exactly.")
        self.play(FadeOut(VGroup(nl, title, true, dmd, edmd, legend, tags)))

    def beat_predict(self):
        self.header("Prediction with the fitted linear models")
        ax = self.ax
        self.play(FadeIn(ax), FadeIn(self.data_mobs[1]))
        par = ax.plot(lambda x: x ** 2, x_range=[-2.1, 2.1], color=C_EIG1, stroke_width=2,
                      stroke_opacity=0.5)
        self.play(Create(par))
        sets = [(self.truth, WHITE, 0.075), (self.pred2, C_BAD, 0.06), (self.pred3, C_GOOD, 0.045)]
        groups = []
        for pts, col, r in sets:
            groups.append(VGroup(polyline(ax, pts, stroke_color=col, stroke_width=2),
                                 *[Dot(ax.c2p(*q), radius=r, color=col) for q in pts]))
        self.play(Create(groups[0][0]), FadeIn(groups[0][1:], lag_ratio=0.1), run_time=2)
        self.say(r"The true orbit (white) follows the parabola.", wait=1.5)
        self.play(Create(groups[1][0]), FadeIn(groups[1][1:], lag_ratio=0.1), run_time=2)
        self.say(r"DMD's linear model in $(x,y)$ (red) cannot bend: it falls to the $x$-axis.")
        self.play(Create(groups[2][0]), FadeIn(groups[2][1:], lag_ratio=0.1), run_time=2)
        self.say(r"EDMD's model in $(x,y,x^2)$ (green) sits on the truth. The dictionary decides.")
        self.say(r"In practice: rich dictionaries (polynomials, kernels, neural nets) and "
                 r"many snapshots.", wait=3)
        self.clear_stage()
        self.drop_all_text()


class KoopmanEDMD(EDMDLesson):
    PROBLEM = dict()


# =========================================================================== pendulum
class PendulumLesson(Lesson):
    DEFAULTS = dict(energies=[-0.85, -0.4, 0.3, 0.85], separatrix=1.0, rotation=1.6,
                    t_show=11.0, dt=0.01, rod=1.25)
    BEATS = ["portrait", "energy", "phase", "spectrum", "recap"]

    def setup_model(self):
        p = self.p
        self.T = {E: M.pendulum_period(E) for E in p["energies"]}
        self.traj = {E: M.pendulum_trajectory(E, p["t_show"], p["dt"]) for E in p["energies"]}
        self.Es = np.linspace(-0.999, 0.999, 200)
        self.Om = np.array([2 * np.pi / M.pendulum_period(E) for E in self.Es])

    def verify(self):
        for E in self.p["energies"]:
            Tn = M.pendulum_period_numeric(E)
            if not np.isclose(self.T[E], Tn, rtol=1e-4):
                raise ValueError(f"period at E={E}: elliptic {self.T[E]:.5f} vs numeric {Tn:.5f}")
            t, th, om = self.traj[E]
            if np.ptp(M.pendulum_energy(th, om)) > 1e-6:
                raise ValueError(f"energy drifts on E={E}")
        if not (abs(self.Om[0] - 1) < 0.01 and self.Om[-1] < 0.5 and np.all(np.diff(self.Om) < 0)):
            raise ValueError("Omega(E) should fall from 1 towards 0")
        print("[verify] periods", {E: round(v, 4) for E, v in self.T.items()},
              "match RK4; energy conserved; Omega(E) decreasing")

    # ---------------------------------------------------------- factories
    def state(self, E, t):
        ts, th, om = self.traj[E]
        return np.interp(t, ts, th), np.interp(t, ts, om)

    def colors(self):
        return [interpolate_color(ManimColor(C_EIG2), ManimColor(C_EIG1), i / 3) for i in range(4)]

    def orbit_curve(self, E, col):
        ts, th, om = M.pendulum_trajectory(E, self.T[E] * 1.002, 0.01)
        ax = self.ax
        return VMobject(stroke_color=col, stroke_width=3).set_points_smoothly(
            [ax.c2p(a, b) for a, b in zip(th[::4], om[::4])])

    # ---------------------------------------------------------- beats
    def beat_portrait(self):
        p = self.p
        self.header(r"A physical example: the pendulum $\ddot\theta=-\sin\theta$")
        ax = self.ax = Axes(x_range=(-PI, PI, PI / 2), y_range=(-2.6, 2.6, 1), x_length=7,
                            y_length=5.4, axis_config=self.AXIS_CONFIG).shift(2.9 * LEFT + 0.3 * DOWN)
        ticks = VGroup(*[MathTex(t, font_size=36).next_to(ax.c2p(v, 0), DOWN, buff=0.45)
                         for t, v in [(r"-\pi", -PI), (r"\pi", PI)]])
        labs = VGroup(MathTex(r"\theta", font_size=32).next_to(ax.x_axis.get_end(), UP, 0.15),
                      MathTex(r"\omega=\dot\theta", font_size=32).next_to(ax.y_axis.get_end(), RIGHT, 0.15))
        cols = self.colors()
        orbits = VGroup(*[self.orbit_curve(E, c) for E, c in zip(p["energies"], cols)])
        sep = VGroup(*[ax.plot(lambda t, s=s: s * np.sqrt(max(2 * (1 + np.cos(t)), 0)),
                               x_range=[-PI, PI], color=GREY_A, stroke_width=2.5) for s in (1, -1)])
        rot = VGroup(*[ax.plot(lambda t, s=s: s * np.sqrt(2 * (p["rotation"] + np.cos(t))),
                               x_range=[-PI, PI], color=GREY_C, stroke_width=2) for s in (1, -1)])
        self.play(Create(ax), FadeIn(ticks), FadeIn(labs))
        self.play(Create(orbits, lag_ratio=0.2), Create(sep), Create(rot), run_time=2.5)
        self.say(r"Phase plane $(\theta,\omega)$: swings are closed loops; outside the separatrix "
                 r"the pendulum spins over the top.")

        pivot = np.array([4.4, 1.9, 0])
        E_out = p["energies"][-1]
        t = ValueTracker(0)
        dots = always_redraw(lambda: VGroup(*[
            Dot(ax.c2p(*self.state(E, t.get_value())), radius=0.08, color=c)
            for E, c in zip(p["energies"], cols)]))

        def bob():
            th, _ = self.state(E_out, t.get_value())
            end = pivot + p["rod"] * np.array([np.sin(th), -np.cos(th), 0])
            return VGroup(Line(pivot, end, color=GREY_A, stroke_width=3),
                          Dot(end, radius=0.13, color=cols[-1]), Dot(pivot, radius=0.05, color=GREY_A))
        pend = always_redraw(bob)
        self.play(FadeIn(dots), FadeIn(pend))
        self.play(t.animate.set_value(p["t_show"]), run_time=p["t_show"], rate_func=linear)
        dots.clear_updaters(); pend.clear_updaters()
        self.say(r"Same law, different energies: the big swing (yellow) takes much longer "
                 r"than the small one.")
        self.play(FadeOut(dots), FadeOut(pend))
        self.portrait = VGroup(ax, ticks, labs, orbits, sep, rot)
        self.orbits_mob, self.cols = orbits, cols

    def beat_energy(self):
        self.header(r"Conserved quantities are eigenfunctions (eigenvalue $1$)")
        rows = VGroup(MathTex(r"\mathcal{K}^t g=g\circ\Phi^t", font_size=34),
                      MathTex(r"H(\theta,\omega)=\tfrac12\omega^2-\cos\theta", font_size=34),
                      MathTex(r"\mathcal{K}^tH=H\quad(\text{eigenvalue }1)", font_size=34).set_color(C_EIG1))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(4.3 * RIGHT + 1.0 * UP)
        for r in rows:
            self.play(Write(r), run_time=0.9)
        self.say(r"In continuous time $\mathcal{K}^t$ composes with the flow $\Phi^t$. "
                 r"The energy $H$ never changes: $\mathcal{K}^tH=H$.")
        self.say(r"Its level sets are exactly the orbits, so any function of $H$ is an "
                 r"eigenfunction too.", wait=2.5)
        self.play(FadeOut(rows))

    def beat_phase(self):
        p = self.p
        self.header(r"The angle variable: an eigenfunction on each orbit")
        c0 = np.array([4.3, 0.9, 0])
        R = 1.35
        circ = Circle(R, color=GREY_B, stroke_width=2).move_to(c0)
        axes2 = VGroup(Line(c0 + 1.6 * LEFT, c0 + 1.6 * RIGHT, color=GREY_D, stroke_width=1.5),
                       Line(c0 + 1.6 * DOWN, c0 + 1.6 * UP, color=GREY_D, stroke_width=1.5))
        clab = MathTex(r"e^{i\alpha}", font_size=30).next_to(circ, UP, buff=0.1).shift(1.3 * RIGHT)
        rows = VGroup(MathTex(r"\varphi_E=e^{i\alpha}\ \ \text{on the orbit }H=E", font_size=30),
                      MathTex(r"\mathcal{K}^t\varphi_E=e^{i\Omega(E)t}\,\varphi_E", font_size=30).set_color(C_EIG1),
                      MathTex(r"\Omega(E)=2\pi/T(E)", font_size=30))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(circ, DOWN, buff=0.5)
        rows.shift((c0[0] - rows.get_center()[0]) * RIGHT)
        pick = [p["energies"][0], p["energies"][-1]]
        pcols = [self.cols[0], self.cols[-1]]
        self.play(self.orbits_mob[1].animate.set_stroke(opacity=0.3),
                  self.orbits_mob[2].animate.set_stroke(opacity=0.3),
                  Create(circ), FadeIn(axes2), FadeIn(clab))
        t = ValueTracker(0)

        def on_circle():
            g = VGroup()
            for E, c in zip(pick, pcols):
                a = 2 * np.pi * t.get_value() / self.T[E]
                q = c0 + R * np.array([np.cos(a), np.sin(a), 0])
                g.add(Line(c0, q, color=c, stroke_width=3), Dot(q, radius=0.09, color=c))
            return g

        dots = always_redraw(lambda: VGroup(*[
            Dot(self.ax.c2p(*self.state(E, t.get_value())), radius=0.09, color=c)
            for E, c in zip(pick, pcols)]))
        hands = always_redraw(on_circle)
        self.play(FadeIn(dots), FadeIn(hands), Write(rows[0]))
        self.say(r"Measure each orbit by the fraction of its period elapsed: the angle $\alpha$.",
                 wait=1.5)
        self.play(t.animate.set_value(p["t_show"]), Write(rows[1]), run_time=p["t_show"],
                  rate_func=linear)
        dots.clear_updaters(); hands.clear_updaters()
        self.play(Write(rows[2]))
        self.say(r"In phase space the big swing lurches, slow at the turning points. "
                 r"$e^{i\alpha}$ turns at a constant rate $\Omega(E)$.")
        self.play(FadeOut(VGroup(circ, axes2, clab, rows, dots, hands)),
                  self.orbits_mob[1].animate.set_stroke(opacity=1),
                  self.orbits_mob[2].animate.set_stroke(opacity=1))

    def beat_spectrum(self):
        p = self.p
        self.header(r"A continuous spectrum")
        self.play(FadeOut(self.portrait))
        ax = Axes(x_range=(-1, 1, 0.5), y_range=(0, 1.25, 0.25), x_length=6.2, y_length=4.6,
                  axis_config=self.AXIS_CONFIG).shift(3.4 * LEFT + 0.1 * DOWN)
        nums = VGroup(*[MathTex(rf"{v:g}", font_size=28).next_to(ax.c2p(v, 0), DOWN, buff=0.2)
                        for v in (-1, -0.5, 0.5, 1)])
        ax.add(nums)
        xl = MathTex(r"E", font_size=32).next_to(ax.x_axis.get_end(), UP, 0.2)
        yl = MathTex(r"\Omega(E)", font_size=32).next_to(ax.y_axis.get_end(), RIGHT, 0.15)
        curve = VMobject(stroke_color=WHITE, stroke_width=4).set_points_smoothly(
            [ax.c2p(E, w) for E, w in zip(self.Es[::4], self.Om[::4])])
        marks = VGroup(*[Dot(ax.c2p(E, 2 * np.pi / self.T[E]), radius=0.08, color=c)
                         for E, c in zip(p["energies"], self.cols)])
        lin = DashedLine(ax.c2p(-1, 1), ax.c2p(1, 1), color=GREY_B, stroke_width=2)
        lin_lab = VGroup(MathTex(r"\Omega=1", font_size=26), Tex("(linear oscillator)", font_size=22))
        lin_lab.arrange(DOWN, buff=0.08).set_color(GREY_B).next_to(ax.c2p(1, 1), RIGHT, 0.2)
        c0, R = np.array([4.3, 0.2, 0]), 1.6
        circ = Circle(R, color=GREY_B, stroke_width=2).move_to(c0)
        cross = VGroup(Line(c0 + 1.8 * LEFT, c0 + 1.8 * RIGHT, color=GREY_D, stroke_width=1.5),
                       Line(c0 + 1.8 * DOWN, c0 + 1.8 * UP, color=GREY_D, stroke_width=1.5))
        arc = Arc(radius=R, start_angle=0, angle=1, arc_center=c0, color=C_EIG1, stroke_width=9)
        on_arc = VGroup(*[Dot(c0 + R * np.array([np.cos(w), np.sin(w), 0]), radius=0.09, color=c)
                          for w, c in zip([2 * np.pi / self.T[E] for E in p["energies"]], self.cols)])
        cl = MathTex(r"e^{i\Omega(E)t},\ t=1", font_size=28).next_to(circ, UP, buff=0.4)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), FadeIn(marks), run_time=1.5)
        self.play(Create(lin), FadeIn(lin_lab))
        self.say(r"Each energy has its own frequency: $\Omega=1$ for tiny swings, $\to0$ at the "
                 r"separatrix. A linear oscillator has only $\Omega=1$.")
        self.play(Create(circ), FadeIn(cross), FadeIn(cl))
        self.play(Create(arc), FadeIn(on_arc), run_time=1.5)
        self.say(r"So the eigenvalues $e^{i\Omega(E)t}$ fill an arc of the unit circle: a "
                 r"continuous spectrum, not three numbers.")
        self.clear_stage()

    def beat_recap(self):
        self.header("Part 1 in four lines")
        rows = self.bullets([
            r"1. $\mathcal{K}g=g\circ F$ is linear on observables, even for nonlinear $F$.",
            r"2. Eigenfunctions are coordinates in which the dynamics is diagonal.",
            r"3. A dictionary plus least squares (EDMD) approximates $\mathcal{K}$ from data.",
            r"4. Spectra: three numbers for the exact map, a continuum for the pendulum.",
        ])
        for r, c in zip(rows, [C_EIG2, C_EIG1, C_GOOD, C_EIG3]):
            r.set_color(c)
        rows.move_to(0.3 * UP)
        for r in rows:
            self.play(FadeIn(r, shift=0.2 * RIGHT))
            self.wait(1.2)
        self.wait(2)
        self.drop_all_text()
        self.play(FadeOut(rows))


class KoopmanPendulum(PendulumLesson):
    PROBLEM = dict()


if __name__ == "__main__":
    check(KoopmanExactMap, KoopmanLift3D, KoopmanEDMD, KoopmanPendulum)
