"""
Part 0 -- the math both topics rest on: eigenvectors, the composition operator g -> g o F,
and concentration of random vectors in high dimension.

    python3 part0_math.py                                       # math check only
    python3 render_check.py part0_math.py MathWarmup --draft    # draft video + contact sheet
"""
from lesson_base import *
import models as M


class WarmupLesson(Lesson):
    DEFAULTS = dict(
        lam=0.9, mu=0.5,
        w=(-1.5, 2.5),                                   # a vector that is not an eigenvector
        starts=[(-1.8, 2.8), (1.8, -2.6), (-1.2, -2.8), (1.5, 3.0), (0.6, 3.2), (-0.5, -3.0)],
        n_iter=14,
        eps=0.25, grid_n=24, g=lambda x: x ** 2, g_tex=r"g(x)=x^2",
        dims=[4, 16, 64, 256, 1024, 10000], n_pairs=20000, seed=0,
    )
    BEATS = ["title", "linear_map", "eigen_coords", "dynamics", "functions",
             "operator_matrix", "concentration", "recap"]

    # ---------------------------------------------------------- pure math
    def setup_model(self):
        p = self.p
        self.A, self.V = M.warmup_matrix(p["lam"], p["mu"])
        self.F = lambda x, y: M.exact_map(x, y, p["lam"], p["mu"])
        self.grid = np.linspace(-1, 1, p["grid_n"])
        self.F1 = lambda x: M.map1d(x, p["eps"])
        self.P = M.composition_matrix(self.grid, self.F1)
        rng = np.random.default_rng(p["seed"])
        self.hists = {D: M.cosine_histogram(D, p["n_pairs"], rng) for D in p["dims"]}
        cs = M.cosine_samples(10000, p["n_pairs"], rng)
        self.frac_2deg = float(np.mean(np.abs(cs) < np.cos(88 * DEGREES)))

    def verify(self):
        p = self.p
        if not np.allclose(self.A @ self.V, self.V @ np.diag([p["lam"], p["mu"]])):
            raise ValueError("A V != V Lambda")
        err = np.max(np.abs(self.P @ p["g"](self.grid) - p["g"](self.F1(self.grid))))
        if err > 5e-3:
            raise ValueError(f"grid Koopman matrix error {err:.2e}")
        for D, (_, f) in self.hists.items():
            if not np.isclose(f.sum(), 1):
                raise ValueError(f"histogram for D={D} does not sum to 1")
        if self.frac_2deg < 0.99:
            raise ValueError(f"only {self.frac_2deg:.3f} of pairs within 2 degrees")
        print(f"[verify] A = {self.A.tolist()}, grid error {err:.1e}, "
              f"within 2 deg at D=10^4: {self.frac_2deg:.4f}")

    # ---------------------------------------------------------- factories
    def make_plane(self):
        plane = NumberPlane(x_range=(-5, 5, 1), y_range=(-3.5, 3.5, 1), x_length=8, y_length=5.6,
                            background_line_style=dict(stroke_color=BLUE_E, stroke_width=1,
                                                       stroke_opacity=0.6))
        return plane.shift(2.6 * LEFT + 0.1 * DOWN)

    def apply_fn(self, mat):
        """Scene-space map q -> o + mat (q - o) about the plane's origin."""
        o, m = self.plane.c2p(0, 0), np.array(mat, dtype=float)
        return lambda q: o + np.append(m @ (q - o)[:2], 0.0)

    def arrow(self, v, color):
        return Arrow(self.plane.c2p(0, 0), self.plane.c2p(*v), buff=0, color=color,
                     stroke_width=5, max_tip_length_to_length_ratio=0.15)

    def right_panel(self, texs, font_size=34, top=2.6):
        col = VGroup(*[MathTex(t, font_size=font_size) for t in texs])
        col.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        if col.width > 5.0:
            col.scale_to_fit_width(5.0)
        col.move_to(4.4 * RIGHT).align_to(top * UP, UP)
        return col

    def hist_bars(self, D):
        centers, frac = self.hists[D]
        ax, w = self.hax, 0.9 * (self.hax.c2p(1 / 32, 0)[0] - self.hax.c2p(0, 0)[0])
        bars = VGroup()
        for c, f in zip(centers, frac):
            h = max(ax.c2p(0, f)[1] - ax.c2p(0, 0)[1], 0.002)
            r = Rectangle(width=w, height=h).move_to(ax.c2p(c, 0), aligned_edge=DOWN)
            bars.add(r.set_fill(C_NEG, 0.85 if f > 0 else 0).set_stroke(width=0))
        return bars

    # ---------------------------------------------------------- beats
    def beat_title(self):
        self.chapter("Part 0", "The math we will use",
                     r"eigenvectors $\cdot$ functions as vectors $\cdot$ high dimensions")

    def beat_linear_map(self):
        p = self.p
        self.header("Linear maps and eigenvectors")
        self.plane = self.make_plane()
        lam, mu = p["lam"], p["mu"]
        v1, v2, w = np.array([3.0, 0]), np.array([2.0, -2.0]), np.array(p["w"])
        self.arrows = VGroup(self.arrow(v1, C_EIG1), self.arrow(v2, C_EIG2), self.arrow(w, WHITE))
        names = [r"v_1", r"v_2", r"w"]
        self.alabels = VGroup(*[
            MathTex(n, font_size=34).set_color(a.get_color()).next_to(a.get_end(), d, buff=0.25)
            for n, a, d in zip(names, self.arrows, [UP, UR, RIGHT])])
        mat = self.right_panel([r"A=\begin{pmatrix}0.9&0.4\\0&0.5\end{pmatrix}"], font_size=40)
        self.play(Create(self.plane, lag_ratio=0.02), run_time=1.5)
        self.play(*[GrowArrow(a) for a in self.arrows], FadeIn(self.alabels), Write(mat))
        self.say(r"A $2\times 2$ matrix moves every vector of the plane. Watch three of them.")

        A = self.A
        new_arrows = VGroup(*[self.arrow(A @ v, a.get_color())
                              for v, a in zip([v1, v2, w], self.arrows)])
        new_labels = VGroup(*[
            MathTex(n, font_size=34).set_color(a.get_color()).next_to(a.get_end(), d, buff=0.25)
            for n, a, d in zip([r"Av_1", r"Av_2", r"Aw"], new_arrows, [UP, UR, RIGHT])])
        self.play(ApplyPointwiseFunction(self.apply_fn(A), self.plane),
                  Transform(self.arrows, new_arrows), Transform(self.alabels, new_labels),
                  run_time=2.5)
        e1 = DashedLine(self.plane.c2p(-5, 0), self.plane.c2p(5, 0), color=C_EIG1,
                        stroke_width=3, dash_length=0.12)
        e2 = DashedLine(self.plane.c2p(-3.5, 3.5), self.plane.c2p(3.5, -3.5), color=C_EIG2,
                        stroke_width=3, dash_length=0.12)
        eqs = self.right_panel([r"A=\begin{pmatrix}0.9&0.4\\0&0.5\end{pmatrix}",
                                rf"A v_1 = {lam:g}\,v_1", rf"A v_2 = {mu:g}\,v_2",
                                r"Aw \not\parallel w"], font_size=40)
        eqs[1].set_color(C_EIG1); eqs[2].set_color(C_EIG2)
        self.play(Create(e1), Create(e2), FadeIn(eqs[1:]), FadeOut(mat), FadeIn(eqs[0]))
        self.eqs, self.eigenlines = eqs, VGroup(e1, e2)
        self.say(r"Eigenvectors stay on their own line and are only scaled: $Av=\lambda v$. "
                 r"The vector $w$ turns.")
        Ai = np.linalg.inv(A)
        back = VGroup(*[self.arrow(v, a.get_color()) for v, a in zip([v1, v2, w], self.arrows)])
        self.play(ApplyPointwiseFunction(self.apply_fn(Ai), self.plane),
                  Transform(self.arrows, back), FadeOut(self.alabels), run_time=1.5)

    def beat_eigen_coords(self):
        self.header("Eigen-coordinates diagonalize")
        pl = self.plane
        diag = VGroup()
        for k in range(-7, 8):                 # lattice lines x + y = k (direction v_2)
            xs = [x for x in (k - 3.5, k + 3.5, -5, 5) if -5 <= x <= 5 and -3.5 <= k - x <= 3.5]
            if len(xs) >= 2 and max(xs) - min(xs) > 1e-6:
                a, b = min(xs), max(xs)
                diag.add(Line(pl.c2p(a, k - a), pl.c2p(b, k - b), color=C_EIG2,
                              stroke_width=1.5, stroke_opacity=0.45))
        panel = self.right_panel([
            r"A = V\Lambda V^{-1}",
            r"V=\begin{pmatrix}1&1\\0&-1\end{pmatrix},\ \Lambda=\begin{pmatrix}0.9&0\\0&0.5\end{pmatrix}",
            r"c = V^{-1}x\ \Rightarrow\ c_{n+1}=\Lambda\,c_n",
            r"c_1(n)=0.9^n\,c_1(0)",
            r"c_2(n)=0.5^n\,c_2(0)"], font_size=34)
        panel[3].set_color(C_EIG1); panel[4].set_color(C_EIG2)
        self.play(FadeOut(self.eqs), Create(diag, lag_ratio=0.05), run_time=1.2)
        for m in panel:
            self.play(Write(m), run_time=0.8)
        self.say(r"In the eigenbasis (the skewed grid) the matrix is just two independent "
                 r"scalings.")
        self.diag, self.panel = diag, panel

    def beat_dynamics(self):
        p = self.p
        self.header(r"Dynamics: iterate $x_{n+1}=Ax_n$")
        pl = self.plane
        self.play(FadeOut(self.arrows), FadeOut(self.panel[:3]))
        dots = VGroup(*[Dot(pl.c2p(*s), radius=0.07, color=WHITE) for s in p["starts"]])
        self.play(FadeIn(dots))
        trails = VGroup()
        pts = [np.array(s, dtype=float) for s in p["starts"]]
        for n in range(p["n_iter"]):
            new = [self.A @ q for q in pts]
            segs = [Line(pl.c2p(*a), pl.c2p(*b), color=GREY_B, stroke_width=2) for a, b in zip(pts, new)]
            trails.add(*segs)
            self.play(*[d.animate.move_to(pl.c2p(*q)) for d, q in zip(dots, new)],
                      *[Create(s) for s in segs], run_time=0.45 if n < 4 else 0.25)
            pts = new
        self.say(r"The $0.5$ direction dies first; then every orbit slides in along the slow "
                 r"eigenline ($0.9$).")

        F_tex = MathTex(r"F(x,y)=\big(0.9\,x,\ \ 0.5\,y+0.31\,x^2\big)", font_size=34)
        F_tex.scale_to_fit_width(4.9).move_to(self.panel[3:]).to_edge(RIGHT, buff=0.35)
        self.play(FadeOut(trails), FadeOut(self.diag), FadeOut(self.eigenlines),
                  FadeOut(self.panel[3:]), FadeIn(F_tex),
                  *[d.animate.move_to(pl.c2p(*s)) for d, s in zip(dots, p["starts"])])
        self.say(r"Now a \emph{nonlinear} map with the same two rates.", wait=1.5)
        trails = VGroup()
        pts = [tuple(s) for s in p["starts"]]
        for n in range(p["n_iter"]):
            new = [self.F(*q) for q in pts]
            segs = [Line(pl.c2p(*a), pl.c2p(*b), color=GREY_B, stroke_width=2) for a, b in zip(pts, new)]
            trails.add(*segs)
            self.play(*[d.animate.move_to(pl.c2p(*q)) for d, q in zip(dots, new)],
                      *[Create(s) for s in segs], run_time=0.45 if n < 4 else 0.25)
            pts = new
        xm = np.sqrt(3.4)
        par = pl.plot(lambda x: x ** 2, x_range=[-xm, xm], color=C_EIG1, stroke_width=4)
        par_lab = MathTex(r"y=x^2", font_size=34).set_color(C_EIG1).next_to(par.get_end(), RIGHT, 0.15)
        self.play(Create(par), FadeIn(par_lab))
        self.say(r"Orbits now hug a parabola: no eigenvectors any more. Part 1 recovers "
                 r"linearity with eigen\emph{functions}.")
        self.clear_stage()

    def beat_functions(self):
        p = self.p
        self.header(r"Functions are vectors too")
        ax = Axes(x_range=(-1, 1, 0.5), y_range=(-0.1, 1.1, 0.5), x_length=6.2, y_length=4.6,
                  axis_config=self.AXIS_CONFIG).shift(3.3 * LEFT + 0.2 * DOWN)
        g, F1 = p["g"], self.F1
        gc = ax.plot(g, x_range=[-1, 1], color=C_EIG1, stroke_width=4)
        kc = ax.plot(lambda x: g(F1(x)), x_range=[-1, 1], color=C_EIG2, stroke_width=4)
        gl = MathTex(p["g_tex"], font_size=34).set_color(C_EIG1)
        kl = MathTex(r"(\mathcal{K}g)(x)=g\big(F(x)\big)", font_size=34).set_color(C_EIG2)

        ins = Axes(x_range=(-1, 1, 1), y_range=(-1, 1, 1), x_length=2.8, y_length=2.8,
                   axis_config=self.AXIS_CONFIG).move_to(4.3 * RIGHT + 1.2 * UP)
        diag = ins.plot(lambda x: x, x_range=[-1, 1], color=GREY_C, stroke_width=2)
        fgr = ins.plot(F1, x_range=[-1, 1], color=WHITE, stroke_width=3)
        flab = MathTex(r"F(x)=x-0.25\sin(\pi x)", font_size=28).next_to(ins, UP, buff=0.15)
        VGroup(gl, kl).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(ins, DOWN, buff=0.35)
        self.play(Create(ax), Create(ins), Create(diag), Create(fgr), FadeIn(flab))
        self.play(Create(gc), FadeIn(gl))
        self.say(r"An observable is any function of the state, for example $g(x)=x^2$. "
                 r"$F$ is a nonlinear map of $[-1,1]$.")
        self.play(TransformFromCopy(gc, kc), FadeIn(kl), run_time=1.5)
        self.say(r"The Koopman operator acts on observables: $\mathcal{K}g=g\circ F$ "
                 r"(measure $g$ one step later).")
        lin = MathTex(r"\mathcal{K}(\alpha g+\beta h)=\alpha\,\mathcal{K}g+\beta\,\mathcal{K}h",
                      font_size=32).next_to(kl, DOWN, buff=0.4).align_to(gl, LEFT)
        if lin.width > 5.2:
            lin.scale_to_fit_width(5.2)
        self.play(Write(lin))
        self.say(r"It is \emph{linear}, even though $F$ is not. The price: the space of "
                 r"functions is infinite-dimensional.")
        self.fax, self.gc, self.kc = ax, gc, kc
        self.inset = VGroup(ins, diag, fgr, flab, lin)
        self.curve_labels = VGroup(gl, kl)

    def beat_operator_matrix(self):
        p = self.p
        self.header(r"On a grid, $\mathcal{K}$ is a matrix")
        ax, xs = self.fax, self.grid
        gv = p["g"](xs)
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.05, color=C_EIG1) for x, y in zip(xs, gv)])
        n = len(xs)
        cell = 3.6 / n
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                if self.P[i, j] > 1e-9:
                    sq = Square(cell).set_stroke(width=0).set_fill(C_EIG2, 0.25 + 0.75 * self.P[i, j])
                    sq.move_to(np.array([(j - (n - 1) / 2) * cell, ((n - 1) / 2 - i) * cell, 0]))
                    cells.add(sq)
        box = Square(3.6).set_stroke(GREY_B, 1.5)
        mat = VGroup(box, cells).move_to(4.3 * RIGHT + 0.75 * DOWN)
        cells.move_to(box)
        mlab = MathTex(rf"P\in\mathbb{{R}}^{{{n}\times{n}}}", font_size=32).next_to(box, UP, 0.2)
        rule = MathTex(r"(Pg)_i=g\big(F(x_i)\big)", font_size=32).next_to(box, DOWN, 0.2)
        self.play(FadeOut(self.inset), self.curve_labels.animate.to_corner(UR, buff=0.4).shift(0.4 * DOWN),
                  FadeIn(dots, lag_ratio=0.05))
        self.say(rf"Sample $g$ at {n} points: the function becomes a vector in "
                 rf"$\mathbb{{R}}^{{{n}}}$.")
        self.play(FadeIn(box), FadeIn(mlab), FadeIn(cells, lag_ratio=0.01), run_time=1.5)
        Pg = self.P @ gv
        new = VGroup(*[Dot(ax.c2p(x, y), radius=0.05, color=C_EIG2) for x, y in zip(xs, Pg)])
        self.play(Write(rule))
        self.play(Transform(dots, new), run_time=2)
        self.say(r"Each row of $P$ interpolates $g$ at $F(x_i)$, and $Pg$ lands on "
                 r"$\mathcal{K}g$. Part 1 builds such matrices from data (EDMD).")
        self.clear_stage()

    def beat_concentration(self):
        p = self.p
        self.header(r"High dimensions: random vectors are almost orthogonal")
        self.hax = ax = Axes(x_range=(-1, 1, 0.5), y_range=(0, 1, 0.25), x_length=8, y_length=4.4,
                             axis_config=self.AXIS_CONFIG).shift(2.2 * LEFT + 0.1 * DOWN)
        ax.add_coordinates()
        xl = MathTex(r"\cos(u,v)", font_size=30).next_to(ax.x_axis.get_end(), DOWN, buff=0.35)
        yl = Tex("fraction of pairs", font_size=26).rotate(PI / 2).next_to(ax.y_axis, LEFT, 0.5)
        dims = p["dims"]
        bars = self.hist_bars(dims[0])

        def readout(D):
            s = 1 / np.sqrt(D)
            ang = np.degrees(np.arcsin(2 * s)) if 2 * s < 1 else 90.0
            r = VGroup(MathTex(rf"D = {D:,}".replace(",", "{,}"), font_size=40),
                       MathTex(rf"\sigma = 1/\sqrt{{D}} = {s:.3g}", font_size=34),
                       MathTex(rf"\pm2\sigma:\ 90^\circ\pm{ang:.1f}^\circ", font_size=34))
            r[0].set_color(C_EIG1)
            return r.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(4.6 * RIGHT + 1.6 * UP)

        rd = readout(dims[0])
        why = VGroup(MathTex(r"u,v\in\{-1,+1\}^D", font_size=30),
                     MathTex(r"u\cdot v=\textstyle\sum_{i=1}^{D}u_iv_i", font_size=30),
                     MathTex(r"u_iv_i=\pm1\ \text{(fair coin)}", font_size=30),
                     MathTex(r"\Rightarrow\ \cos(u,v)\approx\mathcal{N}(0,\,1/D)", font_size=30))
        why.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(4.6 * RIGHT + 1.3 * DOWN)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(FadeIn(bars, lag_ratio=0.01), FadeIn(rd))
        self.say(r"Draw two random $\pm1$ vectors, compute their cosine, repeat 20{,}000 "
                 r"times. Now let $D$ grow.", wait=1.5)
        for D in dims[1:]:
            new_rd = readout(D)
            self.play(Transform(bars, self.hist_bars(D)), FadeOut(rd), FadeIn(new_rd), run_time=1.3)
            rd = new_rd
            self.wait(0.8)
        self.play(FadeIn(why, lag_ratio=0.3), run_time=1.5)
        self.say(r"Each $u_iv_i$ is a fair coin, so $\cos$ has spread $1/\sqrt{D}$ "
                 r"(central limit theorem).")
        pct = np.floor(self.frac_2deg * 1000) / 10
        self.say(rf"At $D=10{{,}}000$, {pct:.1f}\% of random pairs are within $2^\circ$ of "
                 rf"perpendicular. HDC is built on this.")
        self.clear_stage()

    def beat_recap(self):
        self.drop_caption()                    # the last beat's caption does not belong here
        self.header("Takeaways")
        rows = self.bullets([
            r"1. Eigenvectors turn a matrix into independent scalings: $c_i\mapsto\lambda_i c_i$.",
            r"2. $g\mapsto g\circ F$ is linear even for nonlinear $F$ $\Rightarrow$ Koopman (Part 1).",
            r"3. Random high-dimensional vectors are nearly orthogonal $\Rightarrow$ HDC (Part 2).",
        ])
        rows[0].set_color(C_EIG1); rows[1].set_color(C_EIG2); rows[2].set_color(C_POS)
        rows.move_to(0.3 * UP)
        for r in rows:
            self.play(FadeIn(r, shift=0.2 * RIGHT))
            self.wait(1.2)
        self.wait(2)
        self.drop_all_text()
        self.play(FadeOut(rows))


class MathWarmup(WarmupLesson):
    PROBLEM = dict()


if __name__ == "__main__":
    check(MathWarmup)
