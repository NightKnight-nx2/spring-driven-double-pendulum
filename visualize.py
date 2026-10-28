"""
visualize.py -- Yay tahrikli çift sarkaç animasyonu (koyu tema).

Üst panel : fiziksel sistem (yay + M, makaralar, ip/link'ler, m1, m2) ve
            m2'nin son `trail` karelik soluklaşan izi.
Alt panel : kinetik / elastik / gravitasyonel enerjinin yığılmış alan grafiği
            (toplam sabit; grafik zamanla açığa çıkar).

Kullanım:
    python visualize.py                         # rezonans durumu, pencerede oynat
    python visualize.py --case nonresonant
    python visualize.py --save anim.mp4         # ffmpeg gerekir (.gif için Pillow)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import simulate as S  # noqa: E402
from simulate import model  # noqa: E402

# --- Koyu tema ve enerji renkleri -------------------------------------------
BG = "#0d1117"
PANEL = "#111820"
FG = "#c9d1d9"
GRID = "#2a323c"
C_KIN = "#2ecc71"    # kinetik   = yeşil
C_ELA = "#ff9f1a"    # elastik   = turuncu
C_GRA = "#3b82f6"    # gravitasyon = mavi
C_TRAIL = (0.78, 0.55, 1.0)


def spring_points(xc, y0, y1, n=9, amp=0.11, stub=0.10):
    """Düşey zigzag yay noktaları (y0 altta, y1 üstte)."""
    L = y1 - y0
    ys = np.linspace(y0 + stub * L, y1 - stub * L, 2 * n)
    xs = xc + amp * np.where(np.arange(2 * n) % 2 == 0, 1.0, -1.0)
    X = np.r_[xc, xc, xs, xc, xc]
    Y = np.r_[y0, y0 + stub * L, ys, y1 - stub * L, y1]
    return X, Y


def make_animation(res, t_end, fps=60, trail=500, speed=1.0):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Patch, Rectangle

    P = res.P
    p = res.p
    M, m1, m2, k, l1, l2, g = p

    # --- Önceden hesapla (kare ızgarası) -----------------------------------
    dt = speed / fps
    tg = np.arange(0.0, t_end, dt)
    Y = res.sol.sol(tg)
    X1, Y1, X2, Y2 = [np.broadcast_to(a, tg.shape) for a in
                      model.positions(Y[0], Y[1], Y[2], *p)]
    Kg, Ug_el, Ug_gr = res.energies(tg)
    Ten = res.tension(tg)
    Etot = (Kg + Ug_el + Ug_gr)[0]

    # Alt panelde pürüzsüz eğri için daha sık örnekleme
    td = np.linspace(0.0, t_end, 3000)
    Kd, Ud, Gd = res.energies(td)

    # --- Şekil --------------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG})
    fig = plt.figure(figsize=(12, 9.4), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.5], hspace=0.16,
                          left=0.05, right=0.985, top=0.965, bottom=0.075)
    ax = fig.add_subplot(gs[0])
    ae = fig.add_subplot(gs[1])

    # ================= Üst panel: fiziksel sistem =============================
    ax.set_facecolor(PANEL)
    ax.set_xlim(-3.45, 2.75)
    ax.set_ylim(-2.62, 0.78)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)

    xM = -2.05          # M kolonunun x konumu
    yG = 0.0            # makara merkezleri
    yM0 = -1.20         # x = 0 iken M merkezi
    hM, wM = 0.36, 0.55
    ledge_y = -2.42
    R = 0.085

    # duvar, kiriş, çıkıntı (yay tabanı)
    ax.add_patch(Rectangle((-3.45, -2.62), 0.35, 3.40, fc="#1a2028", ec=GRID,
                           hatch="////", lw=0.8, zorder=1))
    ax.add_patch(Rectangle((-3.10, 0.30), 3.34, 0.09, fc="#3a4452", ec="none", zorder=2))
    ax.add_patch(Rectangle((-3.10, ledge_y - 0.09), 1.55, 0.09, fc="#3a4452", ec="none", zorder=2))
    for xc in (xM, 0.0):                                  # makara askıları
        ax.plot([xc, xc], [yG, 0.30], color="#586374", lw=2, zorder=2)
    for xc in (xM, 0.0):                                  # makaralar
        ax.add_patch(Circle((xc, yG), R, fc=BG, ec="#9aa5b4", lw=2, zorder=6))
        ax.plot([xc], [yG], "o", color="#9aa5b4", ms=2.5, zorder=7)

    # hareketli nesneler
    spring_ln, = ax.plot([], [], color=C_ELA, lw=2.2, zorder=3, solid_capstyle="round")
    blockM = Rectangle((xM - wM / 2, yM0 - hM / 2), wM, hM, fc="#7d8a99", ec="#c9d1d9",
                       lw=1.4, zorder=4)
    ax.add_patch(blockM)
    ax.text(xM, yM0, "M", color=BG, ha="center", va="center", fontsize=12,
            fontweight="bold", zorder=5)
    rope_ln, = ax.plot([], [], color="#aeb8c4", lw=1.6, zorder=3)
    rod2_ln, = ax.plot([], [], color="#aeb8c4", lw=2.4, zorder=3, solid_capstyle="round")
    rb1 = 0.045 + 0.05 * m1 ** (1 / 3)
    rb2 = 0.045 + 0.05 * m2 ** (1 / 3)
    bob1 = Circle((0, 0), rb1, fc="#e6edf3", ec="#ffffff", lw=1.2, zorder=6)
    bob2 = Circle((0, 0), rb2, fc="#e6edf3", ec="#ffffff", lw=1.2, zorder=6)
    ax.add_patch(bob1); ax.add_patch(bob2)
    ax.text(0, 0, "", zorder=0)
    lab1 = ax.text(0, 0, "m₁", color=BG, ha="center", va="center", fontsize=9,
                   fontweight="bold", zorder=7)
    lab2 = ax.text(0, 0, "m₂", color=BG, ha="center", va="center", fontsize=9,
                   fontweight="bold", zorder=7)

    # m2'nin izi: soluklaşan LineCollection
    lc = LineCollection([], linewidths=2.0, zorder=2, capstyle="round")
    ax.add_collection(lc)
    ramp = np.linspace(0.0, 1.0, max(trail - 1, 1)) ** 1.6
    base = np.tile(np.array(C_TRAIL + (1.0,)), (len(ramp), 1))
    base[:, 3] = ramp * 0.95

    info = ax.text(1.0, 0.62, "", family="monospace", fontsize=10.5, color=FG,
                   va="top", ha="left", zorder=8)
    ax.text(1.0, -2.40,
            f"M = {M:g} kg   m₁ = {m1:g} kg   m₂ = {m2:g} kg\n"
            f"k = {k:.2f} N/m   l₂ = {l2:g} m\n"
            f"ω_yay = {P['w_x']:.3f} rad/s   ω_yavaş = {P['w_slow']:.3f} rad/s\n"
            f"ω_yay / ω_yavaş = {P['ratio']:.3f}",
            family="monospace", fontsize=8.6, color="#8b949e", va="bottom", ha="left")
    ax.set_title("Yay tahrikli çift sarkaç  —  otoparametrik rezonans" if P["ratio"] > 1.9
                 and P["ratio"] < 2.1 else "Yay tahrikli çift sarkaç",
                 color=FG, fontsize=13, pad=6)

    # ================= Alt panel: yığılmış alan ===============================
    ae.set_facecolor(PANEL)
    ae.stackplot(td, Kd, Ud, Gd, colors=[C_KIN, C_ELA, C_GRA], alpha=0.92, linewidth=0)
    ae.set_xlim(0, t_end)
    ae.set_ylim(0, Etot * 1.04)
    ae.set_xlabel("t [s]")
    ae.set_ylabel("Enerji [J]")
    ae.grid(color=GRID, lw=0.6, alpha=0.7)
    for s in ae.spines.values():
        s.set_color(GRID)
    ae.axhline(Etot, color="#e6edf3", lw=0.9, ls="--", alpha=0.8)
    ae.legend(handles=[Patch(fc=C_KIN, label="Kinetik"),
                       Patch(fc=C_ELA, label="Elastik (yay)"),
                       Patch(fc=C_GRA, label="Gravitasyonel (ofsetli)")],
              loc="lower right", ncol=3, frameon=True, facecolor=BG, edgecolor=GRID,
              fontsize=9, labelcolor=FG)
    ae.set_title("Enerji dağılımı (yığılmış; üst sınır = toplam, sabit)", color=FG,
                 fontsize=11, pad=4)
    # henüz oynatılmamış kısmı örten maske + imleç
    veil = Rectangle((0, 0), t_end, 1, transform=ae.get_xaxis_transform(), fc=PANEL,
                     ec="none", zorder=5)
    ae.add_patch(veil)
    cursor = ae.axvline(0, color="#ffffff", lw=1.2, zorder=6)

    animated = [spring_ln, blockM, rope_ln, rod2_ln, bob1, bob2, lab1, lab2, lc,
                info, veil, cursor]
    for a in animated:
        a.set_animated(True)

    # ---------------------------------------------------------------- update
    def update(i):
        x = Y[0, i]
        yMc = yM0 + x
        blockM.set_xy((xM - wM / 2, yMc - hM / 2))
        sx, sy = spring_points(xM, ledge_y, yMc - hM / 2)
        spring_ln.set_data(sx, sy)
        # ip: M üstü -> sol makara -> pivot makara -> m1
        rope_ln.set_data([xM, xM, 0.0, X1[i]], [yMc + hM / 2, yG, yG, Y1[i]])
        rod2_ln.set_data([X1[i], X2[i]], [Y1[i], Y2[i]])
        bob1.center = (X1[i], Y1[i]); bob2.center = (X2[i], Y2[i])
        lab1.set_position((X1[i], Y1[i])); lab2.set_position((X2[i], Y2[i]))

        j0 = max(0, i - trail + 1)
        px, py = X2[j0:i + 1], Y2[j0:i + 1]
        if len(px) > 1:
            pts = np.column_stack([px, py])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc.set_segments(segs)
            lc.set_color(base[-len(segs):] if len(segs) <= len(base) else base)
        else:
            lc.set_segments([])

        t = tg[i]
        veil.set_x(t); veil.set_width(max(t_end - t, 0))
        cursor.set_xdata([t, t])
        info.set_text(
            f"t        = {t:6.2f} s\n"
            f"x − x_eq = {x - P['x_eq']:+7.3f} m\n"
            f"θ₁       = {np.degrees(Y[1, i]):+7.1f}°\n"
            f"θ₂       = {np.degrees(Y[2, i]):+7.1f}°\n"
            f"T_ip     = {Ten[i]:7.2f} N")
        return animated

    anim = FuncAnimation(fig, update, frames=len(tg), interval=1000 / fps,
                         blit=True, repeat=True)
    return fig, anim, update, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=S.CASES, default="resonant")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--trail", type=int, default=500, help="iz uzunluğu (kare)")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = gerçek zaman")
    ap.add_argument("--save", default=None, help="çıktı dosyası (.mp4 / .gif)")
    ap.add_argument("--png", default=None, help="verilen t[s] anlarından PNG kaydet (virgüllü)")
    a = ap.parse_args()

    import matplotlib
    if a.save or a.png:
        matplotlib.use("Agg")

    print("Simülasyon çözülüyor...")
    res, t_end = S.build_case(a.case)
    print(f"  maks bağıl enerji sapması = {res.rel_dev.max():.2e}")
    fig, anim, update, tg = make_animation(res, t_end, a.fps, a.trail, a.speed)

    if a.png:
        for ts in a.png.split(","):
            i = int(np.argmin(np.abs(tg - float(ts))))
            update(i)
            out = os.path.join(HERE, f"frame_{a.case}_{float(ts):05.1f}s.png")
            fig.savefig(out, dpi=100, facecolor=fig.get_facecolor())
            print("  ", out)
        return
    if a.save:
        if a.save.lower().endswith(".gif"):
            anim.save(a.save, writer="pillow", fps=a.fps, dpi=80,
                      savefig_kwargs={"facecolor": fig.get_facecolor()})
        else:
            anim.save(a.save, writer="ffmpeg", fps=a.fps, dpi=100,
                      savefig_kwargs={"facecolor": fig.get_facecolor()})
        print("Kaydedildi:", a.save)
        return
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
