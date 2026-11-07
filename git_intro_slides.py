"""Build git_intro.pptx: how git and GitHub work, for the students of this course.

    pip install python-pptx matplotlib
    python git_intro_slides.py          # writes git_intro.pptx next to this file

The look (palette, fonts, slide kit) is the same as the weekly decks. Every git
message on a slide is real output (trimmed to the lines that matter), captured on
2026-10-30 with git 2.54 by running the README's student workflow on a copy of this
repository. The merge conflict on slide 10 is the one that week5_todo's hint update
(commit 8fc5b26) caused for anyone who had already filled in the training step.

Colours mean the same thing on every slide:
  blue    main                      (the complete code)
  orange  weekN_todo                (the TODO version, updated by the instructor)
  green   weekN_mywork / your laptop (yours)
  grey    GitHub, history, things git does for you
"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "git_intro.pptx")
REPO_URL = "https://github.com/stanthemaker/tutorial_ml.git"

# ------------------------------------------------------------------ palette
# The weekly decks' palette, plus green for "yours" and red for "destroys work".

INK = RGBColor(0x1C, 0x22, 0x2B)
MUTED = RGBColor(0x6B, 0x75, 0x82)
ACCENT = RGBColor(0xC2, 0x55, 0x1E)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREEN = RGBColor(0x2E, 0x7D, 0x5B)
RED = RGBColor(0xA4, 0x26, 0x2C)
PAPER = RGBColor(0xFA, 0xF9, 0xF7)
RULE = RGBColor(0xD8, 0xD3, 0xCB)
CODE_BG = RGBColor(0xEF, 0xEC, 0xE6)

INK_HEX = "#1c222b"
C_MAIN = "#1f4e79"
C_TODO = "#c2551e"
C_YOURS = "#2e7d5b"
C_GREY = "#6b7582"
C_BAD = "#a4262c"
PAPER_HEX = "#faf9f7"
FILL_GREY = "#ebe8e3"
FILL_GREEN = "#e3eee8"
FILL_BLUE = "#e1e9f1"
FILL_ORANGE = "#f6e4d8"
FILL_RED = "#f4dcdc"

SANS = "Helvetica Neue"      # as in the weekly decks
MONO = "Consolas"            # ships with Office on Windows and Mac (Menlo is Mac-only)
FIG_SANS = "DejaVu Sans"
FIG_MONO = "DejaVu Sans Mono"

W, H = Inches(13.333), Inches(7.5)
M = 0.85                      # side margin, inches
CW = 13.333 - 2 * M           # content width, inches

# ------------------------------------------------------------ figure kit
# Each figure is drawn at the exact size it has on the slide, in inches, so a
# 10 pt label in a figure is 10 pt on the slide.


def fig_canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(PAPER_HEX)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=220, facecolor=PAPER_HEX)
    plt.close(fig)
    return path


def rbox(ax, x, y, w, h, fc, ec="none", lw=1.3, ls="-", r=0.08, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, fs=11, color=INK_HEX, ha="center", va="center", bold=False,
        mono=False, italic=False, z=6):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", style="italic" if italic else "normal",
            family=FIG_MONO if mono else FIG_SANS, zorder=z, linespacing=1.25)


def arrow(ax, a, b, color=C_GREY, lw=1.8, ls="-", rad=0.0, z=2):
    ax.annotate("", xy=b, xytext=a, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=15,
                                connectionstyle=f"arc3,rad={rad}"))


def link(ax, a, b, color="#9aa3ad", lw=2.2, ls="-", z=1):
    """The line between a commit and its parent."""
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="round")


def commit(ax, x, y, color, r=0.15, hollow=False, ring=None, letter=None):
    ax.add_patch(Circle((x, y), r, fc=PAPER_HEX if hollow else color, ec=color,
                        lw=2.0, ls="--" if hollow else "-", zorder=3))
    if ring:
        ax.add_patch(Circle((x, y), r + 0.08, fc="none", ec=ring, lw=2.6, zorder=3))
    if letter:
        txt(ax, x, y, letter, fs=10, color=color if hollow else "white", bold=True)


def tag_width(s, fs=10):
    return len(s) * fs * 0.64 / 72 + 0.22


def tag(ax, x, y, s, color, fs=10, hollow=False, ha="center"):
    """A branch label: a rounded tag with the branch name. Returns (left, right)."""
    w = tag_width(s, fs)
    h = fs * 2.0 / 72
    x0 = {"center": x - w / 2, "left": x, "right": x - w}[ha]
    rbox(ax, x0, y - h / 2, w, h, PAPER_HEX if hollow else color, ec=color, lw=1.5,
         ls="--" if hollow else "-", r=0.06, z=4)
    txt(ax, x0 + w / 2, y, s, fs=fs, color=color if hollow else "white", bold=True)
    return x0, x0 + w


def head(ax, x, y, ha="left"):
    return tag(ax, x, y, "HEAD", INK_HEX, fs=9.5, ha=ha)


def cross(ax, x, y, s=0.11, color=C_BAD):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=2.6, zorder=7)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=2.6, zorder=7)


# ------------------------------------------------------------ slide kit


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = PAPER
    return s


def textbox(slide, x, y, w, h, runs, size=15, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    for s, st in runs:
        add_run(p, s, st, size)
    return box


STYLE_COLOR = {"g": MUTED, "a": ACCENT, "B": BLUE, "O": ACCENT, "Y": GREEN, "R": RED}


def add_run(p, s, st, size):
    r = p.add_run()
    r.text = s
    r.font.size = Pt(size - 1.5 if "m" in st else size)
    r.font.name = MONO if "m" in st else SANS
    r.font.bold = "b" in st
    r.font.color.rgb = next((STYLE_COLOR[c] for c in st if c in STYLE_COLOR), INK)


def para(slide, x, y, w, h, items, size=15, spacing=1.12, gap=7):
    """Paragraphs of mixed runs. An item is a string or a list of (text, style);
    style letters: b bold, m mono, g grey, B blue, O orange, Y green, R red."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = spacing
        p.space_after = Pt(gap)
        for s, st in (it if isinstance(it, list) else [(it, "")]):
            add_run(p, s, st, size)
    return box


def rule(slide, y, x=M, w=CW):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Pt(1.2))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False


def header(slide, kicker, title):
    textbox(slide, M, 0.5, CW, 0.35, [(kicker.upper(), "bO")], size=12)
    textbox(slide, M, 0.85, CW, 0.8, [(title, "bB")], size=29)
    rule(slide, 1.62)


def place(slide, path, x, y, w, h):
    slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))


def code_panel(slide, x, y, w, h, lines, size=12, title=None):
    """Terminal text on a light panel. A line is a string (output, grey) or a tuple:
    ("cmd", command, comment)  ->  $ command   # comment
    ("c", command)             ->  command, no prompt
    ("hl", text)               ->  output that matters, orange
    ("gap",)                   ->  a half-height blank line"""
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.035
    shp.fill.solid()
    shp.fill.fore_color.rgb = CODE_BG
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = tf.margin_bottom = Inches(0.1)
    paras = []

    def new_para():
        p = tf.paragraphs[0] if not paras else tf.add_paragraph()
        paras.append(p)
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.0
        p.space_after = Pt(1)
        p.space_before = Pt(0)
        return p

    def run(p, s, color, bold=False, font=MONO, sz=size):
        r = p.add_run()
        r.text = s
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font

    if title:
        p = new_para()
        run(p, title, MUTED, True, SANS, size + 0.5)
        p.space_after = Pt(4)
    for ln in lines:
        ln = ln if isinstance(ln, tuple) else ("out", ln)
        p = new_para()
        kind = ln[0]
        if kind == "cmd":
            run(p, "$ ", MUTED)
            run(p, ln[1], INK, True)
            if len(ln) > 2 and ln[2]:
                run(p, "   # " + ln[2], MUTED)
        elif kind == "c":
            run(p, ln[1], INK, True)
        elif kind == "gap":
            run(p, " ", INK, sz=size * 0.45)
        else:
            run(p, ln[1], ACCENT if kind == "hl" else MUTED, kind == "hl")
    return shp


def cell(slide, x, y, w, h, runs, size=13):
    return para(slide, x, y, w, h, [runs], size=size, gap=0, spacing=1.05)


# ================================================================ slides
# 1 - 3: what git and GitHub are


def fig_two_places(path):
    w, h = CW, 3.1
    fig, ax = fig_canvas(w, h)
    # GitHub
    rbox(ax, 0.05, 0.08, 4.15, 2.92, FILL_GREY, ec=C_GREY)
    txt(ax, 2.125, 2.68, "GitHub", fs=16, bold=True)
    txt(ax, 2.125, 2.33, "github.com/stanthemaker/tutorial_ml", fs=9.5, mono=True, color=C_GREY)
    txt(ax, 2.125, 2.0, "the shared copy. Its name on your laptop: origin", fs=10)
    tag(ax, 0.65, 1.35, "main", C_MAIN, fs=9.5)
    tag(ax, 1.75, 1.35, "week2_todo", C_TODO, fs=9.5)
    tag(ax, 2.98, 1.35, "week3_todo", C_TODO, fs=9.5)
    tag(ax, 1.2, 0.72, "week4_todo", C_TODO, fs=9.5)
    tag(ax, 2.42, 0.72, "week5_todo", C_TODO, fs=9.5)
    # your laptop
    rbox(ax, 7.43, 0.08, 4.15, 2.92, FILL_GREEN, ec=C_YOURS)
    txt(ax, 9.5, 2.68, "your laptop", fs=16, bold=True)
    txt(ax, 9.5, 2.33, "the folder tutorial_ml/", fs=9.5, mono=True, color=C_GREY)
    txt(ax, 9.5, 2.0, "your files + the full history", fs=10)
    tag(ax, 8.45, 1.35, "main", C_MAIN, fs=9.5)
    tag(ax, 9.75, 1.35, "week5_todo", C_TODO, fs=9.5)
    tag(ax, 9.5, 0.72, "week5_mywork", C_YOURS, fs=9.5)
    txt(ax, 9.5, 0.3, "your own branch: only on your laptop", fs=9, color=C_GREY)
    # what moves between them
    for y, cmd, sub in ((2.3, "git clone", "once, on day 1"), (1.45, "git pull", "each time there is an update")):
        arrow(ax, (4.35, y), (7.28, y), color=INK_HEX, lw=2.2)
        txt(ax, 5.81, y + 0.21, cmd, fs=12, mono=True, bold=True)
        txt(ax, 5.81, y - 0.2, sub, fs=9.5, color=C_GREY)
    arrow(ax, (7.28, 0.6), (4.35, 0.6), color="#b9b3aa", lw=1.8, ls="--")
    cross(ax, 5.81, 0.6)
    txt(ax, 5.81, 0.84, "git push", fs=12, mono=True, bold=True, color=C_GREY)
    txt(ax, 5.81, 0.36, "instructor only", fs=9.5, color=C_BAD)
    return save(fig, path)


def slide_two_places(prs, tmp):
    s = blank(prs)
    header(s, "Git and GitHub", "Git keeps the history; GitHub keeps a shared copy")
    place(s, fig_two_places(os.path.join(tmp, "s01.png")), M, 1.85, CW, 3.1)
    para(s, M, 5.15, 5.75, 2.0, [
        [("Git", "b"), (" is a program on your laptop. It records versions of a folder, so you can "
                        "go back, compare two versions, and combine work.", "")],
        [("GitHub", "b"), (" is a website that keeps a copy of the folder and its history, so "
                           "everyone can download it.", "")],
    ])
    para(s, 6.95, 5.15, 5.5, 2.0, [
        [("In this course you only download: ", "b"),
         ("clone once, then pull each update. You never push.", "")],
        [("Your answers stay on your laptop, on a branch of your own (slide 7).", "g")],
    ])


def fig_commit(path):
    w, h = CW, 2.3
    fig, ax = fig_canvas(w, h)
    commits = [("e63b998", "week5 diffusion"), ("a61c04c", "week5"),
               ("d7188b1", "week5 readme"), ("81f5cc6", "1030 diffusion updates")]
    xs = [0.75, 2.75, 4.75, 6.75]
    y = 1.5
    for a, b in zip(xs, xs[1:]):
        link(ax, (a, y), (b, y))
    for x, (sha, msg) in zip(xs, commits):
        commit(ax, x, y, C_MAIN)
        txt(ax, x, y + 0.34, sha, fs=10, mono=True, bold=True)
        txt(ax, x, y - 0.36, msg, fs=9.5, color=C_GREY)
    arrow(ax, (0.4, 0.62), (7.2, 0.62), color="#b9b3aa", lw=1.3)
    txt(ax, 3.8, 0.4, "older  →  newer", fs=9.5, color=C_GREY)
    # what one commit holds
    rbox(ax, 8.0, 0.32, 3.55, 1.9, FILL_ORANGE, ec=C_TODO)
    rows = [("commit 81f5cc6", True), ("message: 1030 diffusion updates", False),
            ("author + date", False), ("a snapshot of every file", False),
            ("parent: d7188b1, the one before", False)]
    for i, (s, bold) in enumerate(rows):
        txt(ax, 8.17, 1.97 - i * 0.31, s, fs=10 if bold else 9.5, ha="left", mono=bold, bold=bold)
    arrow(ax, (8.0, 1.5), (6.97, 1.5), color=C_TODO, lw=1.6)
    return save(fig, path)


def slide_commit(prs, tmp):
    s = blank(prs)
    header(s, "The basics", "A commit is a saved snapshot of the whole folder")
    place(s, fig_commit(os.path.join(tmp, "s02.png")), M, 1.85, CW, 2.3)
    code_panel(s, M, 4.35, 6.2, 2.4, [
        ("cmd", "git log --oneline -4 81f5cc6", "newest first"),
        "81f5cc6 1030 diffusion updates",
        "d7188b1 week5 readme",
        "a61c04c week5",
        "e63b998 week5 diffusion",
        ("gap",),
        ("cmd", "git show --stat 81f5cc6", "what it changed"),
        "...",
        ("hl", " 55 files changed, 837 insertions(+), 131 deletions(-)"),
    ], size=12)
    para(s, 7.4, 4.3, 5.1, 2.8, [
        [("A commit = ", "b"), ("a snapshot of every file + a message + who and when + a link "
                                "to the commit before it.", "")],
        [("Its id ", "b"), ("(81f5cc6…) is computed from the contents, so every commit has its "
                           "own. The first 7 characters are enough to name it.", "")],
        [("Commits are never edited. ", "b"), ("New work is a new commit on top.", "")],
    ], size=14.5)


def fig_three_places(path):
    w, h = CW, 2.4
    fig, ax = fig_canvas(w, h)
    y0, bh = 0.78, 1.1
    boxes = [(0.1, "working folder", "the files you edit", FILL_GREEN, C_YOURS),
             (4.5, "staging area", "what the next\ncommit will contain", FILL_GREY, C_GREY),
             (8.9, "history", "the commits\n(snapshots)", FILL_BLUE, C_MAIN)]
    for x, title, sub, fc, ec in boxes:
        rbox(ax, x, y0, 2.6, bh, fc, ec=ec)
        txt(ax, x + 1.3, y0 + 0.8, title, fs=13, bold=True)
        txt(ax, x + 1.3, y0 + 0.36, sub, fs=10, color=C_GREY)
    for x0, x1, cmd in ((2.75, 4.45, "git add <file>"), (7.15, 8.85, 'git commit -m "..."')):
        arrow(ax, (x0, y0 + 0.45), (x1, y0 + 0.45), color=INK_HEX, lw=2.2)
        txt(ax, (x0 + x1) / 2, y0 + 0.72, cmd, fs=10.5, mono=True, bold=True)
    # git status looks at the first two
    ax.plot([0.1, 0.1, 7.1, 7.1], [2.0, 2.1, 2.1, 2.0], color=C_GREY, lw=1.2)
    txt(ax, 3.6, 2.25, "git status  shows what is in these two", fs=10, color=C_GREY)
    # restore: throw edits away
    arrow(ax, (10.2, y0 - 0.04), (1.4, y0 - 0.04), color=C_BAD, lw=1.5, ls="--", rad=-0.06)
    txt(ax, 5.8, 0.2, "git restore <file>   throws your edits away: back to the last commit",
        fs=10, color=C_BAD)
    return save(fig, path)


def slide_three_places(prs, tmp):
    s = blank(prs)
    header(s, "The basics", "Three places a change can be")
    place(s, fig_three_places(os.path.join(tmp, "s03.png")), M, 1.85, CW, 2.4)
    code_panel(s, M, 4.4, 7.0, 2.5, [
        ("cmd", "git status"),
        "On branch week5_mywork",
        "Changes not staged for commit:",
        ("hl", "        modified:   week5_diffusion/part1_unet_diffusion.py"),
        ("gap",),
        ("cmd", "git add week5_diffusion/part1_unet_diffusion.py"),
        ("cmd", "git status"),
        "On branch week5_mywork",
        "Changes to be committed:",
        ("hl", "        modified:   week5_diffusion/part1_unet_diffusion.py"),
    ], size=11.5)
    para(s, 8.15, 4.35, 4.35, 2.8, [
        [("git status is always safe: ", "b"), ("it only reads. Run it before add, commit, "
                                                 "checkout and merge.", "")],
        [("Why a staging area: ", "b"), ("you choose which edits go into the next commit.", "")],
        [("The first line, On branch …, says where your next commit will go.", "g")],
    ], size=14.5)


# 4 - 5: branches


def fig_branches(path):
    w, h = CW, 2.9
    fig, ax = fig_canvas(w, h)
    ym, yt, yy = 0.55, 1.45, 2.35
    main = [0.5, 1.6, 4.3]
    link(ax, (main[0], ym), (main[1], ym))
    link(ax, (main[1], ym), (main[2], ym))
    link(ax, (main[1], ym), (2.7, yt))
    link(ax, (2.7, yt), (5.6, yt))
    link(ax, (main[2], ym), (5.6, yt))
    link(ax, (5.6, yt), (7.4, yy))
    link(ax, (7.4, yy), (8.5, yy))
    for x in main:
        commit(ax, x, ym, C_MAIN)
    txt(ax, main[2], ym - 0.33, "new week", fs=9, color=C_GREY)
    tag(ax, main[2] + 0.32, ym, "main", C_MAIN, ha="left")
    commit(ax, 2.7, yt, C_TODO)
    txt(ax, 2.7, yt + 0.33, "# TODO blanks", fs=9, color=C_GREY)
    commit(ax, 5.6, yt, C_TODO)
    txt(ax, 5.8, yt - 0.33, "main merged in", fs=9, color=C_GREY, ha="left")
    tag(ax, 5.92, yt, "week5_todo", C_TODO, ha="left")
    for x, msg in ((7.4, "part 1 done"), (8.5, "part 2 done")):
        commit(ax, x, yy, C_YOURS)
        txt(ax, x, yy - 0.33, msg, fs=9, color=C_GREY)
    _, right = tag(ax, 8.82, yy, "week5_mywork", C_YOURS, ha="left")
    hl, _ = head(ax, right + 0.38, yy)
    arrow(ax, (hl, yy), (right + 0.03, yy), color=INK_HEX, lw=1.5)
    return save(fig, path)


def slide_branches(prs, tmp):
    s = blank(prs)
    header(s, "Branches", "A branch is a label that points at a commit")
    place(s, fig_branches(os.path.join(tmp, "s04.png")), M, 1.85, CW, 2.9)
    code_panel(s, M, 4.95, 3.5, 1.3, [
        ("cmd", "git branch", "* = where you are"),
        "  main",
        ("hl", "* week5_mywork"),
        "  week5_todo",
    ], size=12)
    para(s, 4.7, 4.9, 7.8, 2.3, [
        [("main", "bmB"), ("   the complete code: the answers.", "")],
        [("week5_todo", "bmO"), ("   the same code with key lines blanked out (# TODO). One per week.", "")],
        [("week5_mywork", "bmY"), ("   yours: you make it, and your commits go here.", "")],
        [("A new commit moves only the label you are on (HEAD). git checkout <branch> swaps the "
          "files in your folder to that branch's commit.", "g")],
    ], size=14.5, gap=5)


def fig_two_copies(path):
    w, h = CW, 3.0
    fig, ax = fig_canvas(w, h)
    rbox(ax, 0.05, 0.3, 2.95, 2.55, FILL_GREY, ec=C_GREY)
    txt(ax, 1.52, 2.55, "GitHub", fs=14, bold=True)
    tag(ax, 1.52, 1.75, "main", C_MAIN)
    tag(ax, 1.52, 0.95, "week5_todo", C_TODO)
    rbox(ax, 4.15, 0.05, 7.45, 2.9, FILL_GREEN, ec=C_YOURS)
    txt(ax, 7.88, 2.72, "your laptop", fs=14, bold=True)
    txt(ax, 5.6, 2.33, "copies of GitHub's labels", fs=10.5, bold=True)
    txt(ax, 5.6, 2.1, "move only when you pull", fs=9.5, color=C_GREY)
    txt(ax, 9.95, 2.33, "your branches", fs=10.5, bold=True)
    txt(ax, 9.95, 2.1, "move only when you commit or merge", fs=9.5, color=C_GREY)
    l1, r1 = tag(ax, 5.6, 1.6, "origin/main", C_MAIN, hollow=True)
    l2, r2 = tag(ax, 5.6, 0.85, "origin/week5_todo", C_TODO, hollow=True)
    lm, _ = tag(ax, 9.95, 1.6, "main", C_MAIN)
    ly, _ = tag(ax, 9.95, 0.85, "week5_mywork", C_YOURS)
    _, rt = tag(ax, 9.95, 0.33, "week5_todo", C_TODO, fs=9)
    txt(ax, rt + 0.08, 0.33, "(stays behind)", fs=9, color=C_GREY, ha="left")
    arrow(ax, (3.02, 1.75), (l1 - 0.04, 1.62), color=INK_HEX, lw=2)
    arrow(ax, (3.02, 0.95), (l2 - 0.04, 0.87), color=INK_HEX, lw=2)
    txt(ax, 3.6, 1.38, "fetch", fs=10.5, mono=True, bold=True)
    txt(ax, 3.6, 1.18, "(part of pull)", fs=8.5, color=C_GREY)
    arrow(ax, (r1 + 0.04, 1.6), (lm - 0.04, 1.6), color=C_MAIN, lw=1.8)
    txt(ax, (r1 + lm) / 2, 1.8, "git pull (while on main)", fs=9.5, mono=True)
    arrow(ax, (r2 + 0.04, 0.85), (ly - 0.04, 0.85), color=C_TODO, lw=1.8)
    txt(ax, (r2 + ly) / 2, 1.05, "git merge origin/week5_todo", fs=9.5, mono=True)
    return save(fig, path)


def slide_two_copies(prs, tmp):
    s = blank(prs)
    header(s, "Branches", "GitHub's labels and yours are separate")
    place(s, fig_two_copies(os.path.join(tmp, "s05.png")), M, 1.85, CW, 3.0)
    code_panel(s, M, 5.05, 6.6, 1.5, [
        ("cmd", "git checkout main"),
        ("cmd", "git pull"),
        "From https://github.com/stanthemaker/tutorial_ml",
        ("hl", "   83108d3..567d339  week5_todo -> origin/week5_todo"),
        "Already up to date.",
    ], size=12)
    para(s, 7.75, 5.0, 4.75, 2.2, [
        [("origin/week5_todo", "bmO"), (" is your copy of GitHub's week5_todo. Here GitHub had new "
                                        "TODO commits, so it moved; main had nothing new.", "")],
        [("Your own labels never move by themselves. ", "b"),
         ("Bringing an update into your work is a merge (slide 9).", "")],
    ], size=14)


# 6 - 11: the course workflow


def fig_clone(path):
    w, h = 5.6, 3.25
    fig, ax = fig_canvas(w, h)
    rbox(ax, 0.05, 2.6, 5.5, 0.6, FILL_GREY, ec=C_GREY)
    txt(ax, 2.8, 2.9, "github.com/stanthemaker/tutorial_ml", fs=10.5, mono=True)
    arrow(ax, (2.8, 2.55), (2.8, 2.1), color=INK_HEX, lw=2.2)
    txt(ax, 2.98, 2.33, "git clone", fs=11, mono=True, bold=True, ha="left")
    rbox(ax, 0.05, 0.05, 5.5, 2.0, FILL_GREEN, ec=C_YOURS)
    rows = [("tutorial_ml/", "", True), ("  .git/", "every commit of every branch", False),
            ("  README.md", "the instructions", False),
            ("  week1_… week5_…/", "one folder per week", False),
            ("  data/", "datasets: the scripts download", False),
            ("", "them here, never into git", False)]
    for i, (name, note, bold) in enumerate(rows):
        y = 1.8 - i * 0.285
        txt(ax, 0.22, y, name, fs=10, mono=True, bold=bold, ha="left")
        txt(ax, 2.55, y, note, fs=9.5, color=C_GREY, ha="left")
    return save(fig, path)


def slide_clone(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Day 1: download the repository")
    place(s, fig_clone(os.path.join(tmp, "s06.png")), M, 1.9, 5.6, 3.25)
    code_panel(s, 6.75, 1.9, 5.73, 2.65, [
        ("cmd", "git clone " + REPO_URL),
        "Cloning into 'tutorial_ml'...",
        ("cmd", "cd tutorial_ml"),
        ("cmd", "git branch -a", "every branch"),
        ("hl", "* main"),
        "  remotes/origin/HEAD -> origin/main",
        "  remotes/origin/main",
        "  remotes/origin/week2_todo",
        "  remotes/origin/week3_todo",
        "  remotes/origin/week4_todo",
        "  remotes/origin/week5_todo",
    ], size=11)
    para(s, M, 5.45, CW, 1.8, [
        [("Public repository: ", "b"), ("no account or login needed to clone or pull.", "")],
        [("Windows: ", "b"), ("each week's data entry is a link to the shared data/ folder, and git on "
                              "Windows saves it as a small text file instead. Run every script from the "
                              "tutorial_ml folder, e.g. ", ""),
         ("python week2_mlp/part3_mlp_mnist.py", "m"), (", not from inside week2_mlp/.", "")],
    ], size=14.5)


def fig_new_branch(path):
    w, h = CW, 2.55
    fig, ax = fig_canvas(w, h)
    yc, yt, yh = 1.05, 1.62, 2.18
    panels = [(0.05, "git checkout week5_todo", "switch to the week's TODO branch"),
              (3.95, "git checkout -b week5_mywork", "a new label on the same commit"),
              (7.85, "git commit  (twice)", "only your label moves forward")]
    for x0, cmd, sub in panels:
        txt(ax, x0 + 1.85, 0.42, cmd, fs=11, mono=True, bold=True)
        txt(ax, x0 + 1.85, 0.16, sub, fs=9.5, color=C_GREY)
    for xs in (3.9, 7.8):
        ax.plot([xs, xs], [0.1, 2.45], color="#d8d3cb", lw=1.0)
    # 1: on week5_todo
    x0 = 0.05
    link(ax, (x0 + 0.6, yc), (x0 + 1.85, yc))
    commit(ax, x0 + 0.6, yc, C_TODO)
    commit(ax, x0 + 1.85, yc, C_TODO)
    tag(ax, x0 + 1.85, yt, "week5_todo", C_TODO)
    link(ax, (x0 + 1.85, yt - 0.14), (x0 + 1.85, yc + 0.16), color="#b9b3aa", lw=1.2)
    head(ax, x0 + 1.85, yh, ha="center")
    arrow(ax, (x0 + 1.85, yh - 0.13), (x0 + 1.85, yt + 0.15), color=INK_HEX, lw=1.3)
    # 2: a second label, HEAD moves to it
    x0 = 3.95
    link(ax, (x0 + 0.6, yc), (x0 + 1.85, yc))
    commit(ax, x0 + 0.6, yc, C_TODO)
    commit(ax, x0 + 1.85, yc, C_TODO)
    tag(ax, x0 + 1.25, yt, "week5_todo", C_TODO)
    tag(ax, x0 + 2.75, yt, "week5_mywork", C_YOURS)
    head(ax, x0 + 2.75, yh, ha="center")
    arrow(ax, (x0 + 2.75, yh - 0.13), (x0 + 2.75, yt + 0.15), color=INK_HEX, lw=1.3)
    for tx in (x0 + 1.25, x0 + 2.75):
        link(ax, (tx, yt - 0.14), (x0 + 1.85, yc + 0.16), color="#b9b3aa", lw=1.2)
    # 3: two commits on week5_mywork
    x0 = 7.85
    xs = [x0 + 0.35, x0 + 1.2, x0 + 2.1, x0 + 3.0]
    for a, b in zip(xs, xs[1:]):
        link(ax, (a, yc), (b, yc))
    commit(ax, xs[0], yc, C_TODO)
    commit(ax, xs[1], yc, C_TODO)
    commit(ax, xs[2], yc, C_YOURS)
    commit(ax, xs[3], yc, C_YOURS)
    tag(ax, xs[1], yt, "week5_todo", C_TODO)
    tag(ax, xs[3] - 0.1, yt, "week5_mywork", C_YOURS)
    link(ax, (xs[1], yt - 0.14), (xs[1], yc + 0.16), color="#b9b3aa", lw=1.2)
    link(ax, (xs[3] - 0.1, yt - 0.14), (xs[3], yc + 0.16), color="#b9b3aa", lw=1.2)
    head(ax, xs[3] - 0.1, yh, ha="center")
    arrow(ax, (xs[3] - 0.1, yh - 0.13), (xs[3] - 0.1, yt + 0.15), color=INK_HEX, lw=1.3)
    return save(fig, path)


def slide_new_branch(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Starting a week: make a branch of your own")
    place(s, fig_new_branch(os.path.join(tmp, "s07.png")), M, 1.85, CW, 2.55)
    code_panel(s, M, 4.6, 6.2, 1.45, [
        ("cmd", "git checkout week5_todo"),
        "Switched to a new branch 'week5_todo'",
        "branch 'week5_todo' set up to track 'origin/week5_todo'.",
        ("cmd", "git checkout -b week5_mywork", "-b: create, then switch"),
        ("hl", "Switched to a new branch 'week5_mywork'"),
    ], size=11.5)
    para(s, 7.35, 4.55, 5.15, 2.6, [
        [("Never write answers on week5_todo or main. ", "b"),
         ("Updates arrive on week5_todo. With your work on week5_mywork, an update never "
          "overwrites it: git combines the two when you merge (slide 9).", "")],
        [("Same every week: week2_todo → week2_mywork, week3_todo → week3_mywork, …", "g")],
    ], size=14.5)


def fig_save(path):
    w, h = CW, 2.4
    fig, ax = fig_canvas(w, h)
    # left: commits
    txt(ax, 0.1, 2.22, "each commit moves your label one step", fs=11, bold=True, ha="left")
    y = 1.2
    xs = [0.55, 1.95, 3.35]
    link(ax, (xs[0], y), (xs[1], y))
    link(ax, (xs[1], y), (xs[2], y))
    commit(ax, xs[0], y, C_TODO)
    commit(ax, xs[1], y, C_YOURS)
    commit(ax, xs[2], y, C_YOURS)
    tag(ax, xs[0], y + 0.55, "week5_todo", C_TODO, fs=9)
    link(ax, (xs[0], y + 0.42), (xs[0], y + 0.17), color="#b9b3aa", lw=1.2)
    txt(ax, xs[1], y - 0.36, "add_noise done", fs=9.5, color=C_GREY)
    txt(ax, xs[2], y - 0.36, "training step done", fs=9.5, color=C_GREY)
    _, r = tag(ax, xs[2], y + 0.55, "week5_mywork", C_YOURS)
    hl, _ = head(ax, r + 0.3, y + 0.55)
    arrow(ax, (hl, y + 0.55), (r + 0.03, y + 0.55), color=INK_HEX, lw=1.3)
    link(ax, (xs[2], y + 0.41), (xs[2], y + 0.17), color="#b9b3aa", lw=1.2)
    # right: stash
    ax.plot([5.75, 5.75], [0.1, 2.35], color="#d8d3cb", lw=1.0)
    txt(ax, 6.05, 2.22, "forgot to make your branch first?", fs=11, bold=True, ha="left")
    yb, bh = 0.55, 0.85
    rbox(ax, 6.05, yb, 1.45, bh, FILL_ORANGE, ec=C_TODO)
    txt(ax, 6.775, yb + bh / 2, "your edits\non week5_todo", fs=9.5)
    rbox(ax, 8.2, yb, 1.2, bh, FILL_GREY, ec=C_GREY)
    txt(ax, 8.8, yb + bh / 2, "the stash\n(a shelf)", fs=9.5)
    rbox(ax, 10.1, yb, 1.48, bh, FILL_GREEN, ec=C_YOURS)
    txt(ax, 10.84, yb + bh / 2, "your edits\non week5_mywork", fs=9.5)
    arrow(ax, (7.53, yb + bh / 2), (8.17, yb + bh / 2), color=INK_HEX, lw=1.8)
    arrow(ax, (9.43, yb + bh / 2), (10.07, yb + bh / 2), color=INK_HEX, lw=1.8)
    txt(ax, 7.85, 1.62, "git stash", fs=9.5, mono=True, bold=True)
    txt(ax, 9.75, 1.82, "git checkout -b week5_mywork", fs=9.5, mono=True, bold=True)
    txt(ax, 9.75, 1.6, "git stash pop", fs=9.5, mono=True, bold=True)
    return save(fig, path)


def slide_save(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Saving your work: commit small and often")
    place(s, fig_save(os.path.join(tmp, "s08.png")), M, 1.85, CW, 2.4)
    code_panel(s, M, 4.4, 5.75, 1.3, [
        ("cmd", "git add week5_diffusion/part1_unet_diffusion.py"),
        ("cmd", 'git commit -m "week5: training step done"'),
        ("hl", "[week5_mywork 0f95f08] week5: training step done"),
        " 1 file changed, 3 insertions(+), 5 deletions(-)",
    ], size=11)
    code_panel(s, 6.85, 4.4, 5.63, 1.3, [
        ("cmd", "git stash", "put the edits aside"),
        "Saved working directory and index state WIP on week5_todo",
        ("cmd", "git checkout -b week5_mywork"),
        ("cmd", "git stash pop", "bring them back"),
    ], size=11)
    para(s, M, 5.95, CW, 1.0, [
        [("A good message says what you did: ", ""), ('"week5: training step done"', "m"),
         (", not ", ""), ('"update"', "m"), (".   Windows may print ", "g"),
         ("LF will be replaced by CRLF", "mg"), (": harmless, it is about line endings.", "g")],
    ], size=14)


def fig_update(path):
    w, h = CW, 2.95
    fig, ax = fig_canvas(w, h)
    yy, yu = 2.25, 0.72
    b = (0.9, 1.5)
    y1 = (3.2, yy)
    us = [(2.6, yu), (4.0, yu), (5.4, yu)]
    mg = (7.3, yy)
    link(ax, b, y1)
    link(ax, b, us[0])
    link(ax, us[0], us[1])
    link(ax, us[1], us[2])
    link(ax, y1, mg)
    link(ax, us[2], mg)
    commit(ax, *b, C_TODO)
    txt(ax, b[0], b[1] - 0.34, "83108d3", fs=9.5, mono=True, bold=True)
    txt(ax, b[0], b[1] - 0.55, "where you started", fs=9, color=C_GREY)
    commit(ax, *y1, C_YOURS)
    txt(ax, y1[0], yy + 0.33, "0f95f08", fs=9.5, mono=True, bold=True)
    txt(ax, y1[0] - 0.12, yy - 0.34, "week5: training step done", fs=9, color=C_GREY, ha="left")
    for (x, y), sha, msg in zip(us, ("6f237cd", "8fc5b26", "567d339"),
                                ("todo_v3", "week5_todo_diffusion", "add comments")):
        commit(ax, x, y, C_TODO)
        txt(ax, x, y - 0.34, sha, fs=9.5, mono=True, bold=True)
        txt(ax, x, y - 0.55, msg, fs=9, color=C_GREY)
    tag(ax, 5.75, yu, "origin/week5_todo", C_TODO, hollow=True, ha="left")
    commit(ax, *mg, C_YOURS, r=0.19)
    txt(ax, mg[0], yy + 0.37, "1dc6ac8", fs=9.5, mono=True, bold=True)
    txt(ax, 7.72, yy - 0.38, "the merge: yours + the update", fs=9, color=C_GREY, ha="left")
    _, r = tag(ax, 7.72, yy, "week5_mywork", C_YOURS, ha="left")
    hl, _ = head(ax, r + 0.32, yy)
    arrow(ax, (hl, yy), (r + 0.03, yy), color=INK_HEX, lw=1.3)
    txt(ax, 4.4, 2.72, "your work", fs=10.5, bold=True, color=C_YOURS)
    txt(ax, 3.95, 1.3, "the update (the instructor pushed it to GitHub)", fs=10.5, bold=True, color=C_TODO)
    return save(fig, path)


def slide_update(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Getting updates: pull, then merge into your branch")
    place(s, fig_update(os.path.join(tmp, "s09.png")), M, 1.85, CW, 2.95)
    code_panel(s, M, 4.95, 6.7, 1.6, [
        ("cmd", "git checkout main"),
        ("cmd", "git pull", "download every branch's news"),
        ("hl", "   83108d3..567d339  week5_todo -> origin/week5_todo"),
        ("cmd", "git checkout week5_mywork"),
        ("cmd", "git merge origin/week5_todo", "the update -> your work"),
    ], size=12)
    para(s, 7.85, 4.9, 4.65, 2.3, [
        [("Commit first. ", "b"), ("Git refuses to switch branch if that would overwrite unsaved "
                                   "edits (slide 13).", "")],
        [("The merge is a new commit with two parents. ", "b"), ("Nothing of yours is lost.", "")],
        [("Usually git combines the two by itself. When both changed the same lines, it stops: "
          "next slide.", "g")],
    ], size=14)


CONFLICT = [
    ("ctx", "    t = torch.randint(0, sched.T, (len(x0),), device=device)"),
    ("mark", "<<<<<<< HEAD"),
    ("ours", "    noise = torch.randn_like(x0)"),
    ("ours", "    x_t = sched.add_noise(x0, t, noise)"),
    ("ours", "    loss = F.mse_loss(model(x_t, t, y), noise)"),
    ("mark", "======="),
    ("theirs", "    # TODO: training loop"),
    ("theirs", "    # hint1: torch.randn_like(x0) generates a randnoise tensor …"),
    ("theirs", "    # hint2: get x_t (noisy + digit image) by sched.add_noise(_,_,_)"),
    ("theirs", "    # hint3: pass x_t to model, output a predicted noise, …"),
    ("mark", ">>>>>>> origin/week5_todo"),
    ("ctx", "    opt.zero_grad(set_to_none=True)"),
]


def fig_conflict(path):
    w, h = 7.9, 2.85
    fig, ax = fig_canvas(w, h)
    step = 0.222
    top = h - 0.2
    band = {"ours": FILL_GREEN, "theirs": FILL_ORANGE}
    for i, (kind, s) in enumerate(CONFLICT):
        y = top - i * step
        if kind in band:
            ax.add_patch(plt.Rectangle((0.02, y - step / 2), w - 0.04, step, fc=band[kind], ec="none", zorder=0))
        color = {"ctx": C_GREY, "mark": C_BAD}.get(kind, INK_HEX)
        txt(ax, 0.1, y, s, fs=10, mono=True, ha="left", bold=kind == "mark", color=color)
    y_ours = top - 3 * step
    y_theirs = top - 7.5 * step
    txt(ax, w - 0.1, y_ours, "yours\n(HEAD = week5_mywork)", fs=9.5, bold=True, color=C_YOURS, ha="right")
    txt(ax, w - 0.1, y_theirs, "the update\n(origin/week5_todo)", fs=9.5, bold=True, color=C_TODO, ha="right")
    return save(fig, path)


def slide_conflict(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Merge conflict: both sides changed the same lines")
    code_panel(s, M, 1.85, CW, 0.95, [
        ("cmd", "git merge origin/week5_todo"),
        ("hl", "CONFLICT (content): Merge conflict in week5_diffusion/part1_unet_diffusion.py"),
        "Automatic merge failed; fix conflicts and then commit the result.",
    ], size=12)
    place(s, fig_conflict(os.path.join(tmp, "s10.png")), M, 2.98, 7.9, 2.85)
    para(s, 9.0, 2.9, 3.5, 3.1, [
        [("To finish the merge", "b")],
        [("1. ", "b"), ("Edit the file: keep the lines you want (here: your 3 lines) and delete the "
                        "marker lines ", ""), ("<<<<<<< ======= >>>>>>>", "mR")],
        [("2. ", "b"), ("git add <the file>", "m")],
        [("3. ", "b"), ("git commit", "m"), ("  (git writes the message)", "g")],
        [("Changed your mind?", "b")],
        [("git merge --abort", "m"), (" goes back to before the merge.", "")],
    ], size=13.5, gap=6)
    para(s, M, 6.05, CW, 1.2, [
        [("This happened for real. ", "b"),
         ("On Oct 30 the hints for the training step on week5_todo were rewritten (commit 8fc5b26). "
          "Anyone who had already filled in those lines got exactly this conflict. Keep your 3 lines: "
          "they are what the hints describe.", "g")],
    ], size=14)


DIFF_A = [("ctx", "     def add_noise(self, x0, t, noise):"),
          ("ctx", "         ab = self.alpha_bars[t].view(-1, 1, 1, 1)"),
          ("del", "-        # f ( ab, x0, noise)"),
          ("del", "-        # return"),
          ("add", "+        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise")]
DIFF_B = [("ctx", "     def add_noise(self, x0, t, noise):"),
          ("ctx", "         ab = self.alpha_bars[t].view(-1, 1, 1, 1)"),
          ("del", "-        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise"),
          ("add", "+        return ab.sqrt() * x0 + (1 - ab) * noise")]


def fig_answers(path):
    w, h = CW, 2.45
    fig, ax = fig_canvas(w, h)
    cards = [(0.05, "git diff -w week5_todo main -- week5_diffusion/common.py",
              [("the answer:   ", C_GREY), ("− week5_todo (the TODO)", C_BAD), ("    + main", C_YOURS)],
              DIFF_A, None),
             (5.93, "git diff main -- week5_diffusion/common.py",
              [("your file vs the answer:   ", C_GREY), ("− main", C_BAD), ("    + yours", C_YOURS)],
              DIFF_B, "this student forgot .sqrt(): the diff points at the line")]
    for x0, title, legend, lines, foot in cards:
        rbox(ax, x0, 0.05, 5.65, 2.35, "#f3f1ed", ec="#d8d3cb")
        txt(ax, x0 + 0.15, 2.13, title, fs=10, mono=True, bold=True, ha="left")
        # legend: draw the pieces one after another, each measured where the last ended
        xx = x0 + 0.15
        renderer = fig.canvas.get_renderer()
        for piece, color in legend:
            t = ax.text(xx, 1.86, piece, fontsize=9.5, color=color, ha="left", va="center",
                        family=FIG_SANS, zorder=6)
            xx += t.get_window_extent(renderer=renderer).width / fig.dpi
        for i, (kind, s) in enumerate(lines):
            y = 1.5 - i * 0.24
            if kind in ("add", "del"):
                ax.add_patch(plt.Rectangle((x0 + 0.08, y - 0.12), 5.49, 0.24,
                                           fc=FILL_GREEN if kind == "add" else FILL_RED, ec="none", zorder=1))
            color = {"ctx": C_GREY, "add": C_YOURS, "del": C_BAD}[kind]
            txt(ax, x0 + 0.12, y, s, fs=9.5, mono=True, ha="left", color=color, bold=kind != "ctx")
        if foot:
            txt(ax, x0 + 0.15, 0.25, foot, fs=9.5, color=C_TODO, ha="left", bold=True)
    return save(fig, path)


def slide_answers(prs, tmp):
    s = blank(prs)
    header(s, "The course workflow", "Checking your answer against main")
    place(s, fig_answers(os.path.join(tmp, "s11.png")), M, 1.85, CW, 2.45)
    code_panel(s, M, 4.5, 7.3, 1.35, [
        ("cmd", "git diff -w week5_todo main -- <file>", "the answer"),
        ("cmd", "git diff main -- <file>", "your file vs the answer"),
        ("cmd", "git checkout main", "see everything (commit first)"),
        ("cmd", "git checkout week5_mywork", "back to your work"),
    ], size=12)
    para(s, 8.45, 4.45, 4.05, 2.8, [
        [("−", "bR"), (" lines come from the first name, ", ""), ("+", "bY"), (" lines from the second.", "")],
        [("-w", "bm"), (" ignores changes in spacing. The TODO branch was also reformatted, so some "
                        "lines differ only in layout: the answer is where the # TODO was.", "")],
    ], size=14)


# 12: the website


def fig_url(path):
    w, h = CW, 1.55
    fig, ax = fig_canvas(w, h)
    parts = [("github.com", "the site", C_GREY), ("/stanthemaker", "owner", C_MAIN),
             ("/tutorial_ml", "repository", C_MAIN), ("/tree", "view", C_TODO),
             ("/week5_todo", "branch or commit", C_YOURS)]
    fs = 20
    cw = fs * 0.602 / 72
    total = sum(len(p) for p, _, _ in parts) * cw
    x = (w - total) / 2
    for p, label, color in parts:
        pw = len(p) * cw
        txt(ax, x, 1.12, p, fs=fs, mono=True, ha="left", color=color, bold=True)
        ax.plot([x + 0.06, x + 0.06, x + pw - 0.06, x + pw - 0.06], [0.82, 0.74, 0.74, 0.82],
                color=color, lw=1.4)
        txt(ax, x + pw / 2, 0.5, label, fs=10.5, color=color)
        x += pw
    return save(fig, path)


def slide_github(prs, tmp):
    s = blank(prs)
    header(s, "github.com", "Reading the repository in the browser")
    place(s, fig_url(os.path.join(tmp, "s12.png")), M, 1.8, CW, 1.55)
    cols = [(M, 4.35), (M + 4.35, 3.95), (M + 8.3, 3.33)]
    heads = ["add to github.com/stanthemaker/tutorial_ml", "what you see", "the same with git"]
    top = 3.6
    for (x, w), hd in zip(cols, heads):
        cell(s, x, top, w, 0.4, [(hd, "bg")], size=13)
    rule(s, top + 0.42)
    rows = [("/tree/week5_todo", "the files on one branch", "git checkout week5_todo"),
            ("/commits/main", "the list of commits", "git log main"),
            ("/commit/81f5cc6", "what one commit changed", "git show 81f5cc6"),
            ("/compare/main...week5_todo", "what week5_todo changed: answers out, TODOs in",
             "git diff main...week5_todo")]
    for i, row in enumerate(rows):
        y = top + 0.55 + i * 0.5
        for j, ((x, w), val) in enumerate(zip(cols, row)):
            cell(s, x, y, w, 0.45, [(val, "bmO" if j == 0 else ("m" if j == 2 else ""))], size=14)
    para(s, M, 6.45, CW, 0.8, [
        [("Code → Download ZIP", "b"), (" gives only the files: no history, no branches, no "
                                             "git pull. Use git clone (slide 6).", "")],
    ], size=14.5)


# 13 - 15: when things go wrong


def fig_detached(path):
    w, h = CW, 1.55
    fig, ax = fig_canvas(w, h)
    y = 0.62
    txt(ax, 0.1, 1.3, "normal", fs=11, bold=True, ha="left")
    _, hr = head(ax, 0.3, y)
    tl, tr = tag(ax, hr + 0.45, y, "week5_mywork", C_YOURS, ha="left")
    arrow(ax, (hr + 0.03, y), (tl - 0.03, y), color=INK_HEX, lw=1.3)
    commit(ax, tr + 0.75, y, C_YOURS)
    arrow(ax, (tr + 0.03, y), (tr + 0.58, y), color=INK_HEX, lw=1.3)
    ax.plot([5.6, 5.6], [0.1, 1.45], color="#d8d3cb", lw=1.0)
    txt(ax, 5.9, 1.3, "detached HEAD", fs=11, bold=True, ha="left", color=C_TODO)
    _, hr = head(ax, 6.1, y)
    commit(ax, hr + 0.95, y, C_GREY)
    arrow(ax, (hr + 0.03, y), (hr + 0.78, y), color=INK_HEX, lw=1.3)
    txt(ax, hr + 0.95, y + 0.33, "81f5cc6", fs=9.5, mono=True, bold=True)
    txt(ax, hr + 1.35, y, "no branch label: commits made here\nare easy to lose", fs=10,
        color=C_GREY, ha="left")
    return save(fig, path)


def slide_mistakes(prs, tmp):
    s = blank(prs)
    header(s, "When things go wrong", "Four situations, and what to do")
    place(s, fig_detached(os.path.join(tmp, "s13.png")), M, 1.78, CW, 1.55)
    cols = [(M, 4.1), (M + 4.2, 3.6), (M + 7.9, 3.73)]
    top = 3.45
    for (x, w), hd in zip(cols, ["what you see", "what it means", "what to do"]):
        cell(s, x, top, w, 0.4, [(hd, "bg")], size=13)
    rule(s, top + 0.4)
    rows = [
        ([("error: Your local changes to the following files would be overwritten by checkout", "mO")],
         [("You have unsaved edits, and the branch you are switching to has another version of "
           "that file.", "")],
         [("commit them (slide 8), or ", ""), ("git stash", "m"), (", switch, ", ""), ("git stash pop", "m")]),
        ([("You are in 'detached HEAD' state.", "mO")],
         [("You checked out a commit, not a branch. Fine for looking around.", "")],
         [("git checkout week5_mywork", "m")]),
        ([("On branch week5_todo", "mO"), ("  … and you already committed there", "")],
         [("You forgot git checkout -b. Give your commits their own name, then take a fresh "
           "week5_todo from GitHub.", "")],
         [("git branch -m week5_todo week5_mywork", "m"), ("\n", ""), ("git checkout week5_todo", "m")]),
        ([("a file never shows up in git status", "")],
         [("It matches .gitignore: datasets (data/) and weights (*.pt) are never saved in git.", "")],
         [("nothing to fix; ", ""), ("git check-ignore -v <file>", "m"), (" names the rule", "")]),
    ]
    for i, row in enumerate(rows):
        y = top + 0.5 + i * 0.8
        for (x, w), runs in zip(cols, row):
            cell(s, x, y, w, 0.78, runs, size=13)


def fig_undo(path):
    w, h = CW, 2.95
    fig, ax = fig_canvas(w, h)
    panels = [(0.05, "1. git restore <file>", "safe", C_YOURS),
              (3.97, "2. git revert <commit>", "safe: keeps all history", C_TODO),
              (7.89, "3. git reset --hard <commit>", "destroys work", C_BAD)]
    for x0, title, sub, color in panels:
        txt(ax, x0 + 1.85, 2.72, title, fs=11.5, mono=True, bold=True, color=color)
        txt(ax, x0 + 1.85, 2.45, sub, fs=10, color=color)
    for xs in (3.93, 7.85):
        ax.plot([xs, xs], [0.1, 2.85], color="#d8d3cb", lw=1.0)
    yc = 1.45
    # 1 restore
    x0 = 0.05
    rbox(ax, x0 + 0.2, 1.05, 1.35, 0.8, FILL_RED, ec=C_BAD)
    txt(ax, x0 + 0.875, 1.45, "your edited\nfile", fs=9.5)
    rbox(ax, x0 + 2.15, 1.05, 1.4, 0.8, FILL_BLUE, ec=C_MAIN)
    txt(ax, x0 + 2.85, 1.45, "the file as in\nthe last commit", fs=9.5)
    arrow(ax, (x0 + 1.6, 1.45), (x0 + 2.1, 1.45), color=INK_HEX, lw=1.8)
    txt(ax, x0 + 1.85, 0.55, "loses: your unsaved edits\nto that file", fs=9.5, color=C_GREY)
    # 2 revert
    x0 = 3.97
    xs = [x0 + 0.4, x0 + 1.2, x0 + 2.0, x0 + 2.95]
    for a, b in zip(xs, xs[1:]):
        link(ax, (a, yc), (b, yc))
    for x, l in zip(xs[:3], "ABC"):
        commit(ax, x, yc, C_YOURS, r=0.17, letter=l)
    commit(ax, xs[3], yc, C_TODO, r=0.17, letter="C′")
    txt(ax, xs[3], yc - 0.36, "undoes C", fs=9.5, color=C_TODO, bold=True)
    tag(ax, xs[3] - 0.15, yc + 0.5, "week5_mywork", C_YOURS, fs=9)
    txt(ax, x0 + 1.85, 0.55, "adds a new commit; nothing is deleted.\nSafe on a branch others have.",
        fs=9.5, color=C_GREY)
    # 3 reset
    x0 = 7.89
    xs = [x0 + 0.5, x0 + 1.4, x0 + 2.3]
    link(ax, (xs[0], yc), (xs[1], yc))
    link(ax, (xs[1], yc), (xs[2], yc), ls="--", color="#c9c3ba")
    commit(ax, xs[0], yc, C_YOURS, r=0.17, letter="A")
    commit(ax, xs[1], yc, C_YOURS, r=0.17, letter="B")
    commit(ax, xs[2], yc, "#b9b3aa", r=0.17, hollow=True, letter="C")
    tag(ax, xs[1], yc + 0.5, "week5_mywork", C_YOURS, fs=9)
    arrow(ax, (xs[2] + 0.55, yc + 0.5), (xs[1] + 0.62, yc + 0.5), color=C_BAD, lw=1.5, rad=0.0)
    txt(ax, xs[2] + 0.62, yc + 0.72, "moves back", fs=9, color=C_BAD)
    txt(ax, x0 + 1.85, 0.55, "C and all unsaved edits are gone.\nOnly for your own, unshared commits.",
        fs=9.5, color=C_GREY)
    return save(fig, path)


def slide_undo(prs, tmp):
    s = blank(prs)
    header(s, "When things go wrong", "Undo, from safe to destructive")
    place(s, fig_undo(os.path.join(tmp, "s14.png")), M, 1.8, CW, 2.95)
    xs = [M + 0.05, M + 3.97, M + 7.89]
    code_panel(s, xs[0], 4.9, 3.6, 1.15, [
        ("cmd", "git restore <file>"),
        ("cmd", "git status --short"),
        "(nothing: the edit is gone)",
    ], size=10.5)
    code_panel(s, xs[1], 4.9, 3.75, 1.15, [
        ("cmd", "git revert 0f95f08"),
        ("hl", "[week5_mywork 0e62e7d]"),
        ("hl", ' Revert "week5: training step done"'),
    ], size=10.5)
    code_panel(s, xs[2], 4.9, 3.69, 1.15, [
        ("cmd", "git reset --hard HEAD~1", "1 back"),
        ("hl", "HEAD is now at 0f95f08"),
        ("hl", " week5: training step done"),
    ], size=10.5)
    para(s, M, 6.25, CW, 1.0, [
        [("Before anything risky, keep a backup label: ", "b"), ("git branch backup_week5", "m"),
         (". It costs nothing, and ", ""), ("git checkout backup_week5", "m"),
         (" gets your commits back.", "")],
    ], size=14)


def fig_real(path):
    w, h = CW, 2.8
    fig, ax = fig_canvas(w, h)
    # main's labels sit above its commits and week5_todo's below, so the two lines
    # that join the rows never cross a label
    ym, yt = 1.85, 0.95
    main = [("d7188b1", "week5 readme", 0.75), ("81f5cc6", "full code + slides", 2.15),
            ("cf495ad", "1030_v2: TODO code", 3.55), ("0d5b50e", "revert 1030_v2", 4.95),
            ("340612b", "untrack lock file", 6.35), ("08341e4", "update slides", 7.75)]
    todo = [("7e7a2d3", "week5 todo", 2.15), ("83108d3", "merge main", 7.05), ("6f237cd", "todo_v3", 8.1),
            ("8fc5b26", "", 9.1), ("567d339", "add comments", 10.1)]
    for (_, _, a), (_, _, b) in zip(main, main[1:]):
        link(ax, (a, ym), (b, ym))
    link(ax, (0.75, ym), (2.15, yt))
    for (_, _, a), (_, _, b) in zip(todo, todo[1:]):
        link(ax, (a, yt), (b, yt))
    link(ax, (6.35, ym), (7.05, yt))
    for sha, msg, x in main:
        bad = sha == "cf495ad"
        commit(ax, x, ym, C_MAIN, ring=C_BAD if bad else None)
        txt(ax, x, ym + 0.36, sha, fs=9.5, mono=True, bold=True, color=C_BAD if bad else INK_HEX)
        txt(ax, x, ym + 0.6, msg, fs=9, color=C_BAD if bad else C_GREY, bold=bad)
    tag(ax, 8.1, ym, "main", C_MAIN, ha="left")
    for sha, msg, x in todo:
        commit(ax, x, yt, C_TODO)
        txt(ax, x, yt - 0.34, sha, fs=9.5, mono=True, bold=True)
        if msg:
            txt(ax, x, yt - 0.57, msg, fs=9, color=C_GREY)
    tag(ax, 10.4, yt, "week5_todo", C_TODO, ha="left")
    return save(fig, path)


def slide_real(prs, tmp):
    s = blank(prs)
    header(s, "A real example", "Oct 30: the TODO code landed on main")
    place(s, fig_real(os.path.join(tmp, "s15.png")), M, 1.8, CW, 2.8)
    cw = 3.72
    xs = [M, M + cw + 0.24, M + 2 * (cw + 0.24)]
    para(s, xs[0], 4.75, cw, 1.9, [
        [("What happened", "bB")],
        "1030_v2 was committed on main, but it held the TODO version of the files: main lost its "
        "answers. It was already on GitHub.",
    ], size=13.5, gap=4)
    para(s, xs[1], 4.75, cw, 1.9, [
        [("Why not just delete it", "bB")],
        [("git reset", "m"), (" could remove it, but only with a force-push to GitHub, which rewrites "
                              "history that others had already pulled.", "")],
    ], size=13.5, gap=4)
    para(s, xs[2], 4.75, cw, 0.4, [[("The fix: two new commits", "bB")]], size=13.5)
    code_panel(s, xs[2], 5.15, cw, 1.25, [
        ("cmd", "git checkout main"), ("cmd", "git revert cf495ad"),
        ("cmd", "git checkout week5_todo"), ("cmd", "git merge main"),
    ], size=11)
    para(s, M, 6.6, CW, 0.6, [
        [("Lesson: ", "b"), ("read the first line of git status, ", ""), ("On branch …", "m"),
         (", before every commit.", "")],
    ], size=14.5)


# 16: cheat sheet


def fig_cycle(path):
    w, h = CW, 1.4
    fig, ax = fig_canvas(w, h)
    y, bh = 0.45, 0.62
    boxes = [(0.3, 2.3, "start a week", FILL_ORANGE, C_TODO),
             (3.6, 2.3, "edit + commit", FILL_GREEN, C_YOURS),
             (6.9, 2.5, "pull + merge updates", FILL_BLUE, C_MAIN),
             (9.85, 1.72, "check answers", FILL_GREY, C_GREY)]
    for x, bw, label, fc, ec in boxes:
        rbox(ax, x, y, bw, bh, fc, ec=ec)
        txt(ax, x + bw / 2, y + bh / 2, label, fs=11.5, bold=True)
    arrow(ax, (2.63, y + bh / 2), (3.57, y + bh / 2), color=INK_HEX)
    arrow(ax, (5.93, y + bh / 2 + 0.1), (6.87, y + bh / 2 + 0.1), color=INK_HEX)
    arrow(ax, (6.87, y + bh / 2 - 0.12), (5.93, y + bh / 2 - 0.12), color=INK_HEX)
    arrow(ax, (9.43, y + bh / 2), (9.82, y + bh / 2), color=C_GREY, ls="--")
    txt(ax, 6.4, y + bh + 0.25, "repeat all week", fs=9.5, color=C_GREY)
    return save(fig, path)


def slide_cheatsheet(prs, tmp):
    s = blank(prs)
    header(s, "Cheat sheet", "The commands of this course")
    place(s, fig_cycle(os.path.join(tmp, "s16.png")), M, 1.78, CW, 1.4)
    pw, gap = (CW - 2 * 0.2) / 3, 0.2
    xs = [M, M + pw + gap, M + 2 * (pw + gap)]
    groups = [
        ("Start a week", [("c", "git checkout weekN_todo"), ("c", "git checkout -b weekN_mywork")]),
        ("Save", [("c", "git status"), ("c", "git add <file>"), ("c", 'git commit -m "what you did"')]),
        ("Get updates", [("c", "git checkout main"), ("c", "git pull"), ("c", "git checkout weekN_mywork"),
                         ("c", "git merge origin/weekN_todo")]),
        ("Look (always safe)", [("c", "git status"), ("c", "git log --oneline --graph"), ("c", "git branch -a")]),
        ("Check answers", [("c", "git diff -w weekN_todo main -- <file>"), ("c", "git diff main -- <file>")]),
        ("Fix and undo", [("c", "git stash  /  git stash pop"), ("c", "git merge --abort"),
                          ("c", "git restore <file>"), ("c", "git revert <commit>")]),
    ]
    for i, (title, lines) in enumerate(groups):
        x = xs[i % 3]
        y = 3.45 + (i // 3) * 1.85
        code_panel(s, x, y, pw, 1.7, lines, size=11.5, title=title)


def build():
    tmp = tempfile.mkdtemp()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    for make in (slide_two_places, slide_commit, slide_three_places, slide_branches, slide_two_copies,
                 slide_clone, slide_new_branch, slide_save, slide_update, slide_conflict, slide_answers,
                 slide_github, slide_mistakes, slide_undo, slide_real, slide_cheatsheet):
        make(prs, tmp)
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
