"""
lesson_base.py -- the look and the helpers shared by every chapter, on top of the
skill's pattern_base (beats, verify, hud, caption, formula_column).

    say(text)        bottom caption in text mode ($...$ for math); replaces the last one
    header(text)     small section label, top left; replaces the last one
    chapter(k, t)    full-screen chapter card
    clear_stage()    fade out everything but the header and caption
"""
from pattern_base import *

# palette: one meaning per color across all chapters
C_POS = "#F0A35E"      # +1 component of a hypervector
C_NEG = "#4C8BD9"      # -1 component
C_EIG1 = "#F7D96F"     # slow eigen-direction / first eigenfunction (lambda)
C_EIG2 = "#5CD0B3"     # fast eigen-direction / second eigenfunction (mu)
C_EIG3 = "#C38FFF"     # third eigenvalue (lambda^2)
C_DATA = "#B0B7C3"     # data, grids, neutral
C_BAD = "#FF6B6B"      # a wrong / naive answer
C_GOOD = "#7BD88F"     # the right answer


class Lesson(PatternScene):
    BACKGROUND = "#1b1d23"
    AXIS_CONFIG = dict(color=GREY_B, tip_width=0.2, tip_height=0.2)

    def construct(self):
        self.header_mob = None
        super().construct()

    # ---------------------------------------------------------- text
    def say(self, text, color=WHITE, font_size=32, wait=None):
        """Text-mode caption ($...$ inline math). One at a time, haloed, pinned."""
        new = Tex(text, font_size=font_size).set_color(color)
        if new.width > 13.2:
            new.scale_to_fit_width(13.2)
        new.set_stroke(self.BACKGROUND, width=8, background=True)
        new = self.hud(new.to_edge(DOWN, buff=0.3))
        if self.caption_mob is None:
            self.play(FadeIn(new))
        else:
            self.play(FadeOut(self.caption_mob), FadeIn(new))
        self.caption_mob = new
        if wait is None:                       # reading time: ~ 1 s per 12 characters
            wait = min(4.0, 1.0 + len(text) / 16)
        self.wait(wait)

    def header(self, text):
        new = Tex(text, font_size=30).set_color(GREY_A)
        new = self.hud(new.to_corner(UL, buff=0.3))
        if self.header_mob is None:
            self.play(FadeIn(new, shift=0.2 * RIGHT), run_time=0.6)
        else:
            self.play(FadeOut(self.header_mob), FadeIn(new, shift=0.2 * RIGHT), run_time=0.6)
        self.header_mob = new

    def chapter(self, kicker, title, sub=None):
        k = Tex(kicker, font_size=30).set_color(GREY_B)
        t = Tex(title, font_size=60)
        parts = [k, t]
        if sub:
            parts.append(Tex(sub, font_size=30).set_color(GREY_A))
        card = VGroup(*parts).arrange(DOWN, buff=0.35)
        self.play(FadeIn(k, shift=0.2 * DOWN), Write(t))
        if sub:
            self.play(FadeIn(parts[2]))
        self.wait(1.5)
        self.play(FadeOut(card))

    def clear_stage(self, run_time=0.8):
        keep = {id(self.header_mob), id(self.caption_mob)}
        mobs = [m for m in self.mobjects if id(m) not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def drop_all_text(self):
        outs = [m for m in (self.header_mob, self.caption_mob) if m is not None]
        if outs:
            self.play(*[FadeOut(m) for m in outs], run_time=0.5)
        self.header_mob = self.caption_mob = None

    def bullets(self, lines, font_size=34, buff=0.35, width=12):
        """A left-aligned list laid out once (so later lines never collide)."""
        rows = VGroup(*[Tex(l, font_size=font_size) for l in lines])
        rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
        if rows.width > width:
            rows.scale_to_fit_width(width)
        return rows


class ThreeDLesson(Lesson, ThreeDScene):
    """Lesson with the orbiting camera (layout is not auto-checked: look at frames)."""
