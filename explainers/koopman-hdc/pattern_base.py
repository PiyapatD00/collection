"""
pattern_base.py -- the problem-independent half of the pattern (ManimGX).

    class MyLesson(PatternScene):     # ThreeDPatternScene if the camera leaves the plane
        DEFAULTS = dict(...)          # every knob the scene understands
        BEATS = ["intro", "build", "result"]
        def setup_model(self): ...    # pure math, no mobjects
        def verify(self): ...         # raise if the stated answer is wrong
        def beat_intro(self): ...
        ...

    class Problem1(MyLesson):
        PROBLEM = dict(...)           # only what differs for this problem

    if __name__ == "__main__":        # `python file.py` = math check, no rendering
        check(Problem1)

What this buys you:
  * construct() is a loop over BEATS -> reorder / drop / insert beats per subclass.
  * each beat is a section -> named in `manimgx preview`, a slide in `manimgx present`.
  * self.p = DEFAULTS merged with PROBLEM -> one place to read every setting.
  * verify() runs before the first frame -> wrong math never reaches a render.
  * hud(), caption() -> screen-pinned text that survives 3D camera moves.
  * MANIM_DRAFT=1 -> res() returns coarse surface resolutions for fast previews.
"""
from manimgx import *
import os
import numpy as np

DRAFT = os.environ.get("MANIM_DRAFT", "0") not in ("", "0")
_RES = 0.4 if DRAFT else 1.0


def res(u, v):
    """Surface resolution, reduced in draft mode. Use for every Surface."""
    return (max(4, int(u * _RES)), max(8, int(v * _RES)))


def surface(func, u_range=(0, 1), v_range=(0, 1), resolution=(24, 48),
            color=BLUE, opacity=1.0):
    """A one-color Surface without edge lines. ManimGX's own default is a blue
    checkerboard with thin grey edges; pass resolution=res(u, v)."""
    return Surface(func, u_range=u_range, v_range=v_range, resolution=resolution,
                   checkerboard_colors=False, fill_color=color, fill_opacity=opacity,
                   stroke_width=0)


def integrate(f, a, b, n=20001):
    """Trapezoid rule; good enough to sanity-check a closed-form answer."""
    xs = np.linspace(a, b, n)
    ys = np.array([f(x) for x in xs], dtype=float)
    return float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs)) / 2)


def check(*scene_classes):
    """Math-only preflight. Builds the model and runs verify() without rendering."""
    ok = True
    for cls in scene_classes:
        s = cls.__new__(cls)
        s.p = {**cls.DEFAULTS, **cls.PROBLEM}
        try:
            s.setup_model()
            s.verify()
            print(f"[check] {cls.__name__}: OK")
        except Exception as e:
            ok = False
            print(f"[check] {cls.__name__}: FAILED -> {e}")
    raise SystemExit(0 if ok else 1)


class PatternScene(Scene):
    DEFAULTS = {}
    PROBLEM = {}
    BEATS = []
    # The look these lessons were designed on (ManimGL's): grey background, grey axes
    # with small tips. ManimGX's own is Manim CE's black background and white axes: set
    # BACKGROUND = BLACK and AXIS_CONFIG = {} in a subclass to use it instead.
    BACKGROUND = "#333333"
    AXIS_CONFIG = dict(color=GREY_B, tip_width=0.25, tip_height=0.25)

    # ---------------------------------------------------------- orchestration
    def construct(self):
        self.camera.background_color = self.BACKGROUND
        self.p = {**self.DEFAULTS, **self.PROBLEM}
        self.setup_model()
        self.verify()
        self.caption_mob = None
        for beat in self.BEATS:
            self.next_section(beat)       # a named stop in preview, a slide in present
            getattr(self, "beat_" + beat)()

    # ---------------------------------------------------------- override these
    def setup_model(self):
        """Pure math: functions, ranges, derived numbers. No mobjects here."""

    def verify(self):
        """Raise ValueError if the stated result disagrees with a numeric check."""

    # ---------------------------------------------------------- helpers
    def hud(self, mob):
        """Pin to the screen so camera moves don't affect it. Pinning does not add
        it to the scene: bring it in with FadeIn / Write as usual."""
        self.camera.add_fixed_in_frame_mobjects(mob)
        return mob

    def caption(self, tex, color=WHITE, font_size=36):
        """Single bottom caption; each call replaces the previous one. A halo of
        background color around the glyphs keeps it legible over lines and solids."""
        new = MathTex(tex, font_size=font_size).set_color(color)
        new.set_stroke(self.BACKGROUND, width=8, background=True)
        new = self.hud(new.to_edge(DOWN))
        if self.caption_mob is None:
            self.play(FadeIn(new))
        else:
            # FadeOut + FadeIn, not FadeTransform: FadeTransform animates an unpinned
            # copy of the new caption, which lies flat in 3D until the swap ends.
            self.play(FadeOut(self.caption_mob), FadeIn(new))
        self.caption_mob = new

    def drop_caption(self):
        if self.caption_mob is not None:
            self.play(FadeOut(self.caption_mob), run_time=0.5)
            self.caption_mob = None

    def formula_column(self, texs, answer_tex=None, max_w=6.0, max_h=6.2,
                       font_size=34, edge=RIGHT):
        """Lay out ALL formula lines once (so later lines never collide), pinned.
        Returns (lines, answer). Write them later in any order."""
        lines = [MathTex(t, font_size=font_size) for t in texs]
        col = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        ans = None
        block = VGroup(col)
        if answer_tex:
            ans = MathTex(answer_tex, font_size=int(font_size * 1.6)).set_color(YELLOW)
            block = VGroup(col, ans).arrange(DOWN, buff=0.4)
        if block.width > max_w:
            block.scale_to_fit_width(max_w)
        if block.height > max_h:
            block.scale_to_fit_height(max_h)
        block.to_edge(edge, buff=0.3).to_edge(UP, buff=0.5)
        for m in lines + ([ans] if ans else []):
            self.hud(m)
        return lines, ans


class ThreeDPatternScene(PatternScene, ThreeDScene):
    """PatternScene with a 3D camera: set_camera_orientation, move_camera,
    begin_ambient_camera_rotation. Use it only when the camera must leave the
    plane: `manimgx inspect` checks the layout of 2D scenes, and skips 3D ones."""
