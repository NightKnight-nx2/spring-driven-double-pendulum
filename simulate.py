"""
simulate.py -- Yay tahrikli çift sarkaç: parametre seçimi, ODE çözümü, enerji.

Kullanım:
    python simulate.py --case nonresonant      # enerji korunumu testi (rezonanssız)
    python simulate.py --case resonant         # otoparametrik rezonans (2:1)
    python simulate.py --case resonant --plot  # enerji sapması grafiğini kaydet

Durum vektörü: y = [x, th1, th2, xd, th1d, th2d]
Parametre sırası: (M, m1, m2, k, l1, l2, g)
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Türetilmiş model dosyası yoksa bir kez türet (sonra hep dosyadan okunur)
if not os.path.exists(os.path.join(HERE, "model_generated.py")):
    import derive
    derive.build()
import model_generated as model  # noqa: E402


# ----------------------------------------------------------------------------
# 1) Parametreler ve rezonans koşulu
# ----------------------------------------------------------------------------
def pendulum_modes(m1, m2, L1, L2, g):
    """Çift sarkacın (sabit L1 ile) küçük salınım modları.

    Genelleştirilmiş özdeğer problemi  K v = w^2 Mm v.
    Dönüş: (w_slow, w_fast, v_slow, v_fast)  -- v = (th1, th2) mod şekli.
    """
    Mm = np.array([[(m1 + m2) * L1**2, m2 * L1 * L2],
                   [m2 * L1 * L2,      m2 * L2**2]])
    K = g * np.array([[(m1 + m2) * L1, 0.0],
                      [0.0,            m2 * L2]])
    w2, V = eigh(K, Mm)                 # artan sırada
    w = np.sqrt(w2)
    return w[0], w[1], V[:, 0], V[:, 1]


def make_params(M, m1, m2, L1_eq, l2, g=9.81, ratio=2.0, k=None):
    """Parametreleri üretir; yay frekansı = ratio * (yavaş mod frekansı).

    Küçük salınımda (th = 0) yay modu, sarkacın tüm kütlesini de taşıdığı için
        w_x = sqrt(k / (M + m1 + m2))
    Denge: k x_eq + M g - (m1 + m2) g = 0  ->  x_eq = (m1 + m2 - M) g / k
    Sarkaç, dengede L1_eq = l1 + x_eq uzunluklu bir çift sarkaç gibi salınır.

    ratio=2 -> otoparametrik rezonans (w_x = 2 w_slow). `k` verilirse
    (rezonanssız deneyler için) ratio yok sayılır ve hesaplanan oran döner.
    """
    w_slow, w_fast, v_slow, v_fast = pendulum_modes(m1, m2, L1_eq, l2, g)
    Mtot = M + m1 + m2
    if k is None:
        k = (ratio * w_slow) ** 2 * Mtot
    x_eq = (m1 + m2 - M) * g / k
    l1 = L1_eq - x_eq
    w_x = np.sqrt(k / Mtot)
    return dict(
        p=(M, m1, m2, k, l1, l2, g),
        x_eq=x_eq, L1_eq=L1_eq,
        w_x=w_x, w_slow=w_slow, w_fast=w_fast,
        ratio=w_x / w_slow, v_slow=v_slow, v_fast=v_fast,
    )


def case_nonresonant():
    """Basit, rezonanssız set: w_x / w_slow ≈ 3.3 (2, 1, 1/2 ... uzakta), büyük açılar."""
    P = make_params(M=1.0, m1=1.0, m2=0.5, L1_eq=1.0, l2=0.7, ratio=3.3)
    y0 = np.array([P["x_eq"] + 0.02, 0.5, -0.3, 0.0, 0.0, 0.0])
    return P, y0, 20.0


def case_resonant():
    """Otoparametrik rezonans: w_x = 2 w_slow. Başlangıç: büyük yay genliği,
    sarkaç yavaş mod şeklinde çok küçük bir tohumla başlar."""
    P = make_params(M=1.2, m1=0.5, m2=0.5, L1_eq=1.0, l2=0.5, ratio=2.0)
    v = P["v_slow"] / abs(P["v_slow"][0])
    seed = 0.02
    y0 = np.array([P["x_eq"] + 0.15, seed * v[0], seed * v[1], 0.0, 0.0, 0.0])
    return P, y0, 30.0


CASES = {"nonresonant": case_nonresonant, "resonant": case_resonant}


# ----------------------------------------------------------------------------
# 2) ODE
# ----------------------------------------------------------------------------
def rhs(t, y, p):
    x, th1, th2, xd, th1d, th2d = y
    xdd, a1, a2 = model.accel(x, th1, th2, xd, th1d, th2d, *p)
    return [xd, th1d, th2d, xdd, a1, a2]


@dataclass
class Result:
    P: dict
    sol: object                      # solve_ivp sonucu (dense_output=True)
    t: np.ndarray                    # çözücünün kendi adımları
    y: np.ndarray                    # (6, n)
    K: np.ndarray
    Uel: np.ndarray
    Ug: np.ndarray                   # ham gravitasyonel PE (keyfi referans)
    E: np.ndarray                    # K + Uel + Ug
    E0: float
    E_eq: float                      # dengedeki (x_eq, th=0) potansiyel enerji
    rtol: float
    atol: float
    ug_offset: float = 0.0           # Ug'yi >= 0 yapan ofset (yığılmış grafik için)
    info: dict = field(default_factory=dict)

    # --- yardımcılar -------------------------------------------------------
    @property
    def p(self):
        return self.P["p"]

    @property
    def rel_dev(self):
        """|E - E0| / (E0 - E_eq): dengeye göre 'uyarılma enerjisine' bağıl sapma."""
        return np.abs(self.E - self.E0) / (self.E0 - self.E_eq)

    def energies(self, t):
        """Verilen zamanlarda (K, Uel, Ug_shifted) -- yığılmış grafik için."""
        Y = self.sol.sol(t)
        K, Uel, Ug = energy_components(Y, self.p)
        return K, Uel, Ug - self.ug_offset

    def tension(self, t):
        """İp gerginliği T = M (xdd + g) + k x  (M kütlesinin hareket denkleminden)."""
        Y = self.sol.sol(np.atleast_1d(t))
        M, m1, m2, k, l1, l2, g = self.p
        xdd = np.array([model.accel(*Y[:, i], *self.p)[0] for i in range(Y.shape[1])])
        return M * (xdd + g) + k * Y[0]


def energy_components(Y, p):
    x, th1, th2, xd, th1d, th2d = Y
    K = model.kinetic(x, th1, th2, xd, th1d, th2d, *p)
    Uel = model.elastic(x, th1, th2, *p)
    Ug = model.gravity(x, th1, th2, *p)
    return K, Uel, Ug


def run(P, y0, t_end, rtol=1e-10, atol=1e-10, dev_tol=1e-6, verbose=True):
    """RK45 ile çöz; enerji sapması dev_tol'u aşarsa toleransları sıkılaştır."""
    p = P["p"]
    zero = np.zeros(6)
    zero[0] = P["x_eq"]
    _, _, Ug_eq = energy_components(zero, p)
    Uel_eq = model.elastic(P["x_eq"], 0.0, 0.0, *p)
    E_eq = Ug_eq + Uel_eq

    while True:
        sol = solve_ivp(rhs, (0.0, t_end), y0, method="RK45", args=(p,),
                        rtol=rtol, atol=atol, dense_output=True)
        K, Uel, Ug = energy_components(sol.y, p)
        E = K + Uel + Ug
        res = Result(P=P, sol=sol, t=sol.t, y=sol.y, K=K, Uel=Uel, Ug=Ug, E=E,
                     E0=E[0], E_eq=E_eq, rtol=rtol, atol=atol)
        dev = res.rel_dev.max()
        if verbose:
            print(f"  rtol=atol={rtol:.0e}: {sol.t.size} adım, "
                  f"maks. bağıl enerji sapması = {dev:.3e}")
        if dev <= dev_tol or rtol <= 1e-13:
            break
        rtol *= 0.1
        atol *= 0.1
        if verbose:
            print(f"  sapma > {dev_tol:.0e}: toleranslar sıkılaştırılıyor")

    # Yığılmış alan grafiği için Ug >= 0 olacak ofset (yoğun çıktı üzerinden)
    tt = np.linspace(0, t_end, 20001)
    Ug_dense = energy_components(sol.sol(tt), p)[2]
    res.ug_offset = Ug_dense.min()
    return res


def report(res, t_end):
    P = res.P
    M, m1, m2, k, l1, l2, g = P["p"]
    print("Parametreler:")
    print(f"  M={M}, m1={m1}, m2={m2}, l2={l2}, g={g}")
    print(f"  k={k:.4f} N/m, l1={l1:.5f} m (x=0'daki üst link), "
          f"x_eq={P['x_eq']:.5f} m, L1_eq={P['L1_eq']:.3f} m")
    print(f"  w_x={P['w_x']:.4f}, w_slow={P['w_slow']:.4f}, w_fast={P['w_fast']:.4f} rad/s"
          f"   ->  w_x / w_slow = {P['ratio']:.4f},  w_x / w_fast = {P['w_x']/P['w_fast']:.4f}")
    dev = res.rel_dev
    print("Enerji korunumu:")
    print(f"  E0 = {res.E0:.8f} J,  E0 - E_denge = {res.E0 - res.E_eq:.8f} J")
    print(f"  maks |E - E0|            = {np.abs(res.E - res.E0).max():.3e} J")
    print(f"  maks bağıl sapma         = {dev.max():.3e}   "
          f"({'TAMAM' if dev.max() < 1e-6 else 'BÜYÜK'}, hedef < 1e-6)")
    tt = np.linspace(0, t_end, 6001)
    T = res.tension(tt)
    print(f"İp gerginliği: min = {T.min():.4f} N, maks = {T.max():.4f} N "
          f"({'ip gergin kalıyor' if T.min() > 0 else 'DİKKAT: ip gevşiyor, model geçersiz!'})")
    Y = res.sol.sol(tt)
    print(f"Genlikler: |x - x_eq| maks = {np.abs(Y[0]-P['x_eq']).max():.4f} m, "
          f"|th1| maks = {np.abs(Y[1]).max():.4f}, |th2| maks = {np.abs(Y[2]).max():.4f} rad")


def plot_energy(res, t_end, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(res.t, res.E - res.E0, color="#ff8a3d", lw=0.8)
    ax[0].set_ylabel("E(t) - E0  [J]")
    ax[0].set_title(f"Enerji sapması (RK45, rtol=atol={res.rtol:.0e})")
    ax[1].semilogy(res.t, np.maximum(res.rel_dev, 1e-18), color="#4aa3ff", lw=0.8)
    ax[1].axhline(1e-6, color="#888", ls="--", lw=0.8, label="1e-6 hedefi")
    ax[1].set_ylabel("bağıl sapma"); ax[1].set_xlabel("t [s]"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"  grafik: {path}")


def build_case(name):
    """visualize.py için: (Result, t_end)."""
    P, y0, t_end = CASES[name]()
    return run(P, y0, t_end, verbose=False), t_end


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES, default="nonresonant")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    P, y0, t_end = CASES[a.case]()
    print(f"== Durum: {a.case}, t_son = {t_end} s ==")
    res = run(P, y0, t_end)
    report(res, t_end)
    if a.plot:
        plot_energy(res, t_end, os.path.join(HERE, f"energy_dev_{a.case}.png"))
