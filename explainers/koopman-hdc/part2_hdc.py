"""
Part 2 -- hyperdimensional computing (vector symbolic architectures), then a bridge back
to Koopman. Every number on screen is computed at D = 10,000; the grids show the first
64 components.

    python3 part2_hdc.py                                    # math check only
    python3 render_check.py part2_hdc.py HDCBasics --draft
"""
from lesson_base import *
import models as M


class HDCLesson(Lesson):
    DEFAULTS = dict(
        D=10000, seed=7, side=8, noise=0.3,
        fillers=["red", "green", "blue", "circle", "square", "small", "large"],
        record=[("COLOR", "red"), ("SHAPE", "circle"), ("SIZE", "small")],
        queries=["COLOR", "SHAPE"],
        strip=16,
    )
    BEATS = ["title", "hypervectors", "similarity", "binding", "bundling", "record",
             "sequence", "bridge"]

    # ---------------------------------------------------------- pure math
    def setup_model(self):
        p = self.p
        rng = np.random.default_rng(p["seed"])
        D = p["D"]
        self.v = {k: M.hv(D, rng) for k in ["a", "b", "c", "d"]}
        self.roles = {r: M.hv(D, rng) for r, _ in p["record"]}
        self.items = {f: M.hv(D, rng) for f in p["fillers"]}
        a, b, c, d = (self.v[k] for k in "abcd")
        self.noisy = M.flip(a, p["noise"], rng)
        self.ab = M.bind(a, b)
        self.unbound = M.bind(self.ab, b)
        self.s = M.bundle(a, b, c)
        self.R = M.bundle(*[M.bind(self.roles[r], self.items[f]) for r, f in p["record"]])
        self.answers = {}
        for q in p["queries"]:
            probe = M.bind(self.R, self.roles[q])
            sims = {f: M.cos(probe, v) for f, v in self.items.items()}
            self.answers[q] = (probe, sims, max(sims, key=sims.get))
        self.abc = M.bind(M.permute(a, 2), M.permute(b, 1), c)
        self.cba = M.bind(M.permute(c, 2), M.permute(b, 1), a)
        self.cos = M.cos

    def verify(self):
        p, a, b, c, d = self.p, *(self.v[k] for k in "abcd")
        tol = 4 / np.sqrt(p["D"])
        if not np.array_equal(self.unbound, a):
            raise ValueError("unbinding did not return a")
        if abs(M.cos(self.ab, a)) > tol or abs(M.cos(self.ab, b)) > tol:
            raise ValueError("a (x) b is not dissimilar to its factors")
        if not np.isclose(M.cos(self.noisy, a), 1 - 2 * p["noise"]):
            raise ValueError("noise level")
        if any(abs(M.cos(self.s, v) - 0.5) > tol for v in (a, b, c)) or abs(M.cos(self.s, d)) > tol:
            raise ValueError("bundle similarities are not ~0.5 / ~0")
        truth = dict(p["record"])
        for q, (_, sims, best) in self.answers.items():
            if best != truth[q] or abs(sims[best] - 0.5) > tol:
                raise ValueError(f"query {q} returned {best} ({sims[best]:.3f})")
        if abs(M.cos(self.abc, self.cba)) > tol or abs(M.cos(M.permute(a), a)) > tol:
            raise ValueError("permutation does not decorrelate")
        print(f"[verify] D={p['D']}: unbind exact, cos(a(x)b, a)={M.cos(self.ab, a):+.3f}, "
              f"bundle {[round(M.cos(self.s, v), 3) for v in (a, b, c, d)]}, "
              f"queries {[(q, x[2], round(x[1][x[2]], 3)) for q, x in self.answers.items()]}, "
              f"cos(abc, cba)={M.cos(self.abc, self.cba):+.3f}")

    # ---------------------------------------------------------- factories
    def grid(self, v, cell=0.28, n=None):
        n = n or self.p["side"]
        g = VGroup(*[Square(cell).set_stroke(width=0).set_fill(C_POS if x > 0 else C_NEG, 1)
                     for x in v[:n * n]])
        return g.arrange_in_grid(rows=n, cols=n, buff=0.04 * cell / 0.28)

    def labeled(self, v, tex, cell=0.28, color=WHITE, font_size=34):
        g = self.grid(v, cell)
        lab = MathTex(tex, font_size=font_size).set_color(color).next_to(g, DOWN, buff=0.2)
        return VGroup(g, lab)

    def legend(self):
        items = VGroup(*[VGroup(Square(0.22).set_stroke(width=0).set_fill(c, 1),
                                MathTex(t, font_size=28)).arrange(RIGHT, buff=0.15)
                         for c, t in [(C_POS, "+1"), (C_NEG, "-1")]])
        return items.arrange(RIGHT, buff=0.5).to_corner(UR, buff=0.4)

    def sim_bars(self, pairs, width=4.0, highlight=None, font_size=30):
        """Rows: label (right-aligned) | bar of length |cos| | signed value."""
        labs = [MathTex(n, font_size=font_size) for n, _ in pairs]
        lw = max(l.width for l in labs)
        rows = VGroup()
        for i, ((name, val), lab) in enumerate(zip(pairs, labs)):
            y = -0.48 * i
            lab.move_to([lw, y, 0], aligned_edge=RIGHT)
            bar = Rectangle(width=max(abs(val), 0.004) * width, height=0.28)
            bar.set_fill(C_GOOD if name == highlight else C_DATA, 0.9).set_stroke(width=0)
            bar.move_to([lw + 0.25, y, 0], aligned_edge=LEFT)
            num = MathTex(f"{val:+.3f}", font_size=font_size - 2).set_color(GREY_A)
            num.move_to([lw + 0.4 + width, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(lab, bar, num))
        return rows

    # ---------------------------------------------------------- beats
    def beat_title(self):
        self.chapter("Part 2", "Hyperdimensional computing",
                     r"computing with random high-dimensional vectors")

    def beat_hypervectors(self):
        self.header(r"Hypervectors: random vectors in $\{-1,+1\}^D$, $D=10{,}000$")
        trio = VGroup(*[self.labeled(self.v[k], rf"\mathbf{{{k}}}") for k in "abc"])
        trio.arrange(RIGHT, buff=1.4).shift(0.5 * UP)
        leg = self.legend()
        self.play(FadeIn(leg))
        for t in trio:
            self.play(FadeIn(t[0], lag_ratio=0.01), FadeIn(t[1]), run_time=1)
        self.say(r"Each symbol gets a random hypervector. We show the first 64 of its "
                 r"10{,}000 components.")
        self.say(r"Information is spread over all components (holographic): no single "
                 r"component means anything.", wait=3)
        self.trio, self.leg = trio, leg

    def beat_similarity(self):
        a, b, c = (self.v[k] for k in "abc")
        self.header(r"Similarity: $\cos(\mathbf{u},\mathbf{v})=\mathbf{u}\cdot\mathbf{v}/D$")
        cs = MathTex(rf"\cos(\mathbf a,\mathbf b)={self.cos(a, b):+.3f},\quad"
                     rf"\cos(\mathbf a,\mathbf c)={self.cos(a, c):+.3f},\quad"
                     rf"\cos(\mathbf b,\mathbf c)={self.cos(b, c):+.3f}", font_size=34)
        cs.next_to(self.trio, DOWN, buff=0.5)
        self.play(Write(cs))
        self.say(r"Random hypervectors are quasi-orthogonal: $|\cos|\approx1/\sqrt{D}=0.01$, "
                 r"as in Part 0.")
        # noise: flip 30% of a's components
        n = self.p["side"] ** 2
        src = self.trio[0]
        ghost = self.labeled(self.noisy, r"\mathbf a'", color=C_EIG1).move_to(src)
        self.play(FadeOut(self.trio[1:]), FadeOut(cs),
                  src.animate.move_to(5.0 * LEFT + 0.5 * UP), ghost.animate.move_to(2.0 * LEFT + 0.5 * UP))
        flips = [i for i in range(n) if self.noisy[i] != a[i]]
        self.say(rf"Corrupt $\mathbf a$: flip {int(self.p['noise'] * 100)}\% of its components "
                 r"at random.", wait=1)
        self.play(*[ghost[0][i].animate.set_fill(C_POS if self.noisy[i] > 0 else C_NEG)
                    for i in flips], lag_ratio=0.05, run_time=1.5)
        bars = self.sim_bars([(r"\cos(\mathbf a',\mathbf a)", self.cos(self.noisy, a)),
                              (r"\cos(\mathbf a',\mathbf b)", self.cos(self.noisy, b)),
                              (r"\cos(\mathbf a',\mathbf c)", self.cos(self.noisy, c))],
                             width=3.0, highlight=r"\cos(\mathbf a',\mathbf a)")
        bars.next_to(ghost, RIGHT, buff=0.7)
        self.play(FadeIn(bars, lag_ratio=0.2))
        self.say(r"Still $\cos=0.40$ to $\mathbf a$: forty standard deviations above chance. "
                 r"Hypervectors tolerate heavy noise.")
        self.clear_stage()

    def beat_binding(self):
        a, b = self.v["a"], self.v["b"]
        self.header(r"Binding $\otimes$: elementwise product")
        A = self.labeled(a, r"\mathbf a")
        B = self.labeled(b, r"\mathbf b")
        C = self.labeled(self.ab, r"\mathbf a\otimes\mathbf b", color=C_EIG1)
        row = VGroup(A, B, C).arrange(RIGHT, buff=1.6).shift(0.6 * UP)
        ops = VGroup(MathTex(r"\otimes", font_size=56).move_to((A[0].get_right() + B[0].get_left()) / 2),
                     MathTex(r"=", font_size=64).move_to((B[0].get_right() + C[0].get_left()) / 2))
        self.play(FadeIn(A), FadeIn(B), FadeIn(ops[0]), FadeIn(self.leg))
        self.play(FadeIn(ops[1]), FadeIn(C[1]),
                  LaggedStart(*[FadeIn(sq) for sq in C[0]], lag_ratio=0.03), run_time=2)
        self.say(r"Multiply component by component: $(+1)(-1)=-1$, $(-1)(-1)=+1$.", wait=2)
        info = MathTex(rf"\cos(\mathbf a\otimes\mathbf b,\mathbf a)={self.cos(self.ab, a):+.3f},\quad"
                       rf"\cos(\mathbf a\otimes\mathbf b,\mathbf b)={self.cos(self.ab, b):+.3f}",
                       font_size=32).next_to(row, DOWN, buff=0.45)
        self.play(Write(info))
        self.say(r"The result is dissimilar to both factors: binding makes a \emph{new} symbol, "
                 r"like a key--value pair.")
        # unbinding
        bb = self.labeled(M.bind(b, b), r"\mathbf b\otimes\mathbf b=\mathbf 1")
        U = self.labeled(self.unbound, r"(\mathbf a\otimes\mathbf b)\otimes\mathbf b", color=C_GOOD)
        row2 = VGroup(C.copy(), B.copy(), U).arrange(RIGHT, buff=1.6).move_to(row)
        ops2 = VGroup(MathTex(r"\otimes", font_size=56).move_to((row2[0][0].get_right() + row2[1][0].get_left()) / 2),
                      MathTex(r"=", font_size=64).move_to((row2[1][0].get_right() + row2[2][0].get_left()) / 2))
        self.play(FadeOut(info), FadeOut(A), FadeOut(ops), C.animate.move_to(row2[0]),
                  B.animate.move_to(row2[1]), FadeIn(ops2))
        self.play(FadeIn(U[1]), LaggedStart(*[FadeIn(sq) for sq in U[0]], lag_ratio=0.03), run_time=2)
        eq = MathTex(r"(\mathbf a\otimes\mathbf b)\otimes\mathbf b=\mathbf a\otimes(\mathbf b\otimes\mathbf b)"
                     r"=\mathbf a\quad(\text{exactly})", font_size=32).next_to(row2, DOWN, buff=0.45)
        self.play(Write(eq))
        self.say(r"Bind with $\mathbf b$ again and $\mathbf a$ comes back exactly: "
                 r"$\mathbf b\otimes\mathbf b$ is all ones.")
        la = MathTex(r"\mathbf x\mapsto\mathbf x\otimes\mathbf b\ =\ \mathrm{diag}(\mathbf b)\,\mathbf x,"
                     r"\qquad \mathrm{diag}(\mathbf b)^2=I", font_size=32).next_to(eq, DOWN, buff=0.3)
        self.play(Write(la))
        self.say(r"Linear algebra view: $\mathrm{diag}(\mathbf b)$ is orthogonal and its own "
                 r"inverse, so binding preserves every cosine.")
        self.clear_stage()

    def beat_bundling(self):
        a, b, c, d = (self.v[k] for k in "abcd")
        self.header(r"Bundling $+$: majority vote")
        small = VGroup(*[self.labeled(self.v[k], rf"\mathbf{{{k}}}", cell=0.16) for k in "abc"])
        small.arrange(DOWN, buff=0.2).to_edge(LEFT, buff=1.0).shift(0.15 * DOWN)
        S = self.labeled(self.s, r"\mathbf s=[\mathbf a+\mathbf b+\mathbf c]", color=C_EIG1)
        S.move_to(1.3 * LEFT + 0.4 * UP)
        arrow = Arrow(small.get_right() + 0.2 * RIGHT, S[0].get_left() + 0.2 * LEFT, buff=0.1,
                      color=GREY_B, stroke_width=3)
        self.play(FadeIn(small, lag_ratio=0.1), FadeIn(self.leg))
        self.play(GrowArrow(arrow), FadeIn(S[1]),
                  LaggedStart(*[FadeIn(sq) for sq in S[0]], lag_ratio=0.03), run_time=2)
        self.say(r"Add the vectors and keep the sign of each component, a majority vote. "
                 r"$[\,\cdot\,]$ means $\mathrm{sign}$.")
        bars = self.sim_bars([(rf"\cos(\mathbf s,\mathbf {k})", self.cos(self.s, self.v[k]))
                              for k in "abcd"], width=3.2)
        for r in bars[:3]:
            r[1].set_fill(C_GOOD)
        bars.next_to(S, RIGHT, buff=0.8)
        self.play(FadeIn(bars, lag_ratio=0.2))
        why = MathTex(r"P(\text{agree with }\mathbf a)=\tfrac34\ \Rightarrow\ \cos=\tfrac34-\tfrac14=\tfrac12",
                      font_size=30).next_to(bars, DOWN, buff=0.5)
        self.fit_to_frame(why)
        self.play(Write(why))
        self.say(r"The bundle stays similar to each member ($\approx0.5$) and unrelated to "
                 r"$\mathbf d$: it represents the \emph{set}.")
        self.clear_stage()

    def fit_to_frame(self, mob, margin=0.3):
        right = config.frame_width / 2 - margin
        if mob.get_right()[0] > right:
            mob.shift((right - mob.get_right()[0]) * RIGHT)
        return mob

    def beat_record(self):
        p = self.p
        self.header("A record: bind roles to fillers, bundle the pairs")
        terms = r"+".join(rf"\mathrm{{{r}}}\otimes\mathrm{{{f}}}" for r, f in p["record"])
        rec = MathTex(rf"\mathbf R=\big[\,{terms}\,\big]", font_size=34).to_edge(UP, buff=1.0)
        self.play(Write(rec))
        self.say(r"Encode an object (a small red circle) as one hypervector: role $\otimes$ "
                 r"filler, then bundle.")
        q = p["queries"][0]
        others = [(r, f) for r, f in p["record"] if r != q]
        noise = r"+".join(rf"\mathrm{{{q}}}\otimes\mathrm{{{r}}}\otimes\mathrm{{{f}}}" for r, f in others)
        unb = MathTex(rf"\mathbf R\otimes\mathrm{{{q}}}=\big[\,\mathrm{{{dict(p['record'])[q]}}}+"
                      rf"\underbrace{{{noise}}}_{{\text{{quasi-orthogonal noise}}}}\,\big]",
                      font_size=30).next_to(rec, DOWN, buff=0.4)
        if unb.width > 13:
            unb.scale_to_fit_width(13)
        self.play(Write(unb))
        self.say(rf"Query: bind $\mathbf R$ with $\mathrm{{{q}}}$. The matching filler survives; "
                 r"the other pairs turn into noise.")

        def bars_for(qq):
            _, sims, best = self.answers[qq]
            br = self.sim_bars([(rf"\mathrm{{{f}}}", sims[f]) for f in p["fillers"]],
                               width=4.5, highlight=rf"\mathrm{{{best}}}", font_size=28)
            return br.next_to(unb, DOWN, buff=0.45).set_x(-1.6)

        bars = bars_for(q)
        title = VGroup(*[Tex(t, font_size=28) for t in
                         ["cleanup memory:", "cosine to every", "stored item"]])
        title.arrange(DOWN, aligned_edge=LEFT, buff=0.12).set_color(GREY_A)
        title.next_to(bars, RIGHT, buff=0.6).align_to(bars, UP)
        self.play(FadeIn(bars, lag_ratio=0.1), FadeIn(title), run_time=1.5)
        best = self.answers[q][2]
        self.say(rf"Compare with every stored item and take the nearest: the answer is "
                 rf"$\mathrm{{{best}}}$ ($\cos\approx0.5$; all others $\approx0$).")
        q2 = p["queries"][1]
        others2 = [(r, f) for r, f in p["record"] if r != q2]
        noise2 = r"+".join(rf"\mathrm{{{q2}}}\otimes\mathrm{{{r}}}\otimes\mathrm{{{f}}}" for r, f in others2)
        unb2 = MathTex(rf"\mathbf R\otimes\mathrm{{{q2}}}=\big[\,\mathrm{{{dict(p['record'])[q2]}}}+"
                       rf"\underbrace{{{noise2}}}_{{\text{{quasi-orthogonal noise}}}}\,\big]",
                       font_size=30).move_to(unb)
        if unb2.width > 13:
            unb2.scale_to_fit_width(13)
        bars2 = bars_for(q2)
        self.play(FadeOut(unb), FadeIn(unb2), Transform(bars, bars2), run_time=1.5)
        self.say(rf"Ask $\mathrm{{{q2}}}$ instead and $\mathrm{{{self.answers[q2][2]}}}$ wins. "
                 r"One vector, queried like a data structure.")
        self.clear_stage()

    def beat_sequence(self):
        a = self.v["a"]
        k = self.p["strip"]
        self.header(r"Permutation $\rho$: encoding order")
        cell = 0.5
        strip = VGroup(*[Square(cell).set_stroke(width=0).set_fill(C_POS if x > 0 else C_NEG, 1)
                         for x in a[:k]]).arrange(RIGHT, buff=0.06).shift(1.6 * UP)
        lab = MathTex(r"\mathbf a", font_size=34).next_to(strip, UP, buff=0.3).align_to(strip, LEFT)
        self.play(FadeIn(strip, lag_ratio=0.03), FadeIn(lab), FadeIn(self.leg))
        step = cell + 0.06
        new = Square(cell).set_stroke(width=0).set_fill(C_POS if a[-1] > 0 else C_NEG, 1)
        new.move_to(strip[0])
        self.play(strip.animate.shift(step * RIGHT), FadeIn(new, shift=step * RIGHT),
                  run_time=1.2)
        self.play(FadeOut(strip[-1]), Transform(lab, MathTex(r"\rho(\mathbf a)", font_size=34)
                                                .move_to(lab, aligned_edge=LEFT)))
        self.say(r"$\rho$ shifts every component one place, cyclically: a permutation matrix, "
                 r"orthogonal, with $\rho^D=I$.")
        rows = VGroup(
            MathTex(rf"\cos(\rho(\mathbf a),\mathbf a)={self.cos(M.permute(a), a):+.3f}", font_size=32),
            MathTex(r"\text{abc}\ \mapsto\ \rho^2(\mathbf a)\otimes\rho(\mathbf b)\otimes\mathbf c", font_size=32),
            MathTex(r"\text{cba}\ \mapsto\ \rho^2(\mathbf c)\otimes\rho(\mathbf b)\otimes\mathbf a", font_size=32),
            MathTex(rf"\cos(\text{{abc}},\text{{cba}})={self.cos(self.abc, self.cba):+.3f}", font_size=32),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(strip, DOWN, buff=0.7)
        rows[3].set_color(C_EIG1)
        for r in rows:
            self.play(Write(r), run_time=0.9)
        self.say(r"The power of $\rho$ marks the position, so the same letters in another order "
                 r"give an unrelated vector.")
        self.say(r"Bundle the n-grams of a text and you get a language profile, as in the "
                 r"language-recognition project in this collection.", wait=3)
        self.clear_stage()
        self.play(FadeOut(self.leg))

    def beat_bridge(self):
        self.header("Koopman and HDC: two ways to lift")
        left = VGroup(
            Tex("Koopman", font_size=40).set_color(C_EIG2),
            MathTex(r"p\ \mapsto\ \Psi(p)\in\mathbb{R}^k", font_size=32),
            Tex(r"lift \emph{designed} so that $\mathcal{K}$ acts linearly", font_size=28),
            Tex(r"eigenfunctions diagonalize the dynamics", font_size=28),
            Tex(r"goal: predict and analyze dynamics", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        right = VGroup(
            Tex("HDC", font_size=40).set_color(C_POS),
            MathTex(r"\text{symbol}\ \mapsto\ \mathbf v\in\{-1,+1\}^D", font_size=32),
            Tex(r"lift chosen \emph{at random}: quasi-orthogonal", font_size=28),
            Tex(r"$\otimes,\ +,\ \rho$ compose structure", font_size=28),
            Tex(r"goal: represent, store and query", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        cols = VGroup(left, right).arrange(RIGHT, buff=1.4, aligned_edge=UP).shift(0.6 * UP)
        if cols.width > 13:
            cols.scale_to_fit_width(13)
        div = Line(UP, DOWN, color=GREY_D).scale_to_fit_height(cols.height + 0.3).move_to(cols)
        div.set_x((left.get_right()[0] + right.get_left()[0]) / 2)
        common = Tex(r"Common thread: move to a bigger space where linear algebra does the work.",
                     font_size=32).set_color(C_EIG1).next_to(cols, DOWN, buff=0.6)
        self.play(FadeIn(left, lag_ratio=0.15), run_time=2)
        self.play(Create(div), FadeIn(right, lag_ratio=0.15), run_time=2)
        self.say(r"Both lift. Koopman's lift is engineered (an invariant subspace); HDC's is "
                 r"random (concentration of measure).")
        self.play(Write(common))
        self.wait(3)
        self.drop_all_text()
        self.play(FadeOut(VGroup(cols, div, common)))
        end = Tex("Thanks for watching", font_size=48)
        self.play(FadeIn(end))
        self.wait(1.5)
        self.play(FadeOut(end))


class HDCBasics(HDCLesson):
    PROBLEM = dict()


if __name__ == "__main__":
    check(HDCBasics)
