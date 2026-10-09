"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-5(b): identify and label each fixed point on the phase
portrait, and mark the corresponding x-locations on the potential.

Same representative potential as part (a):

    V'(x) = x (x - 2)(x - 3)(x - 4)

Fixed points are labelled A-D, left to right, on both panels.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})


def V(x):
    return x**5 / 5 - 9 * x**4 / 4 + 26 * x**3 / 3 - 12 * x**2


# label, x, type
FP = [('A', 0.0, 'saddle'),
      ('B', 2.0, 'centre'),
      ('C', 3.0, 'saddle'),
      ('D', 4.0, 'centre')]

CENTRE_C = 'tab:blue'
SADDLE_C = 'tab:red'


def style(kind):
    """marker, facecolour, edgecolour for a fixed point of this type."""
    if kind == 'centre':
        return 'o', CENTRE_C, CENTRE_C
    return 's', 'white', SADDLE_C


fig, (ax_v, ax_p) = plt.subplots(
    2, 1, figsize=(8.4, 8.0), sharex=True,
    gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.10})

xs = np.linspace(-0.75, 5.15, 1200)

# ---------------- top panel: the potential ---------------------------
ax_v.plot(xs, V(xs), color='tab:purple', lw=2.4, zorder=2)

for lab, x0, kind in FP:
    m, fc, ec = style(kind)
    y0 = V(x0)
    ax_v.plot(x0, y0, m, mfc=fc, mec=ec, mew=2.0, ms=10, zorder=6)
    # letter in a boxed annotation just above the marker
    ax_v.annotate(lab, xy=(x0, y0), xytext=(0, 17),
                  textcoords='offset points', ha='center', va='bottom',
                  fontsize=12, fontweight='bold', color=ec, zorder=7,
                  bbox=dict(boxstyle='round,pad=0.28', fc='white',
                            ec=ec, lw=1.3))
    # vertical guide down through both panels
    ax_v.axvline(x0, color='0.8', lw=1.0, ls='-', zorder=0)
    ax_p.axvline(x0, color='0.8', lw=1.0, ls='-', zorder=0)

ax_v.plot([], [], 'o', color=CENTRE_C, ms=10, label='minimum of $V$')
ax_v.plot([], [], 's', mfc='white', mec=SADDLE_C, mew=2.0, ms=10,
          label='maximum of $V$')
ax_v.set_ylabel('$V(x)$')
ax_v.set_title('Potential field, with the $x$-location of each fixed point')
ax_v.set_ylim(-11.5, 5.5)
ax_v.legend(loc='lower right', framealpha=0.95)
for sp in ('top', 'right'):
    ax_v.spines[sp].set_visible(False)

# ---------------- bottom panel: the phase portrait --------------------
xg = np.linspace(-0.75, 5.15, 900)
vg = np.linspace(-5.0, 5.0, 900)
X, Vv = np.meshgrid(xg, vg)
E = Vv**2 / 2 + V(X)

levels = [-8.40, -8.00, V(3.0), -5.0, -2.0, V(0.0), 2.5]
cols = ['0.5', '0.5', SADDLE_C, '0.5', '0.5', SADDLE_C, '0.5']
lws = [1.0, 1.0, 2.1, 1.0, 1.0, 2.1, 1.0]
for lev, c, lw in zip(levels, cols, lws):
    ax_p.contour(X, Vv, E, levels=[lev], colors=[c], linewidths=lw)

for lab, x0, kind in FP:
    m, fc, ec = style(kind)
    ax_p.plot(x0, 0.0, m, mfc=fc, mec=ec, mew=2.0, ms=10, zorder=6)
    ax_p.annotate(lab, xy=(x0, 0.0), xytext=(0, 20),
                  textcoords='offset points', ha='center', va='bottom',
                  fontsize=12, fontweight='bold', color=ec, zorder=7,
                  bbox=dict(boxstyle='round,pad=0.28', fc='white',
                            ec=ec, lw=1.3))

ax_p.plot([], [], color=SADDLE_C, lw=2.1, label='separatrices')
ax_p.axhline(0, color='0.8', lw=0.9, zorder=0)
ax_p.set_xlabel('$x$')
ax_p.set_ylabel('$v$')
ax_p.set_xlim(-0.75, 5.15)
ax_p.set_ylim(-5.0, 5.0)
ax_p.set_xticks([x for _, x, _ in FP])
ax_p.set_xticklabels([f'$x_{{{lab}}}$' for lab, _, _ in FP])
ax_p.set_title('Phase portrait, with each fixed point labelled')
ax_p.legend(loc='upper left', framealpha=0.95)

fig.savefig('HW1_5b_labelled.png', dpi=200, bbox_inches='tight')

# ---------------- table for the writeup -------------------------------
print(f'{"label":>6} {"x":>6} {"V(x)":>10}  {"portrait":>9}  '
      f'{"potential":>14}  stability')
print('-' * 68)
for lab, x0, kind in FP:
    pot = 'local minimum' if kind == 'centre' else 'local maximum'
    stab = 'neutrally stable' if kind == 'centre' else 'unstable'
    print(f'{lab:>6} {x0:>6.1f} {V(x0):>10.3f}  {kind:>9}  '
          f'{pot:>14}  {stab}')

plt.show()