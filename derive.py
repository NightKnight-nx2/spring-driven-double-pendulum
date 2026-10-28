"""
derive.py -- Yay tahrikli çift sarkaç: sympy ile Lagrange türetmesi.

Sistem (şematik):

      ___ makara (O, sarkacın asıldığı nokta)
     |   \
     |    \  r = l1 + x   (değişken uzunluklu üst ip/link)
     M     m1
     |       \  l2
   yay        m2
     |
   (zemin/duvar dibi)

* M kütlesi düşey yayın (sabit k, doğal boy) üstünde durur; ip M'nin üstünden
  makaraya çıkar ve diğer ucu çift sarkacın üst linkini (m1'e kadar) oluşturur.
* x  : yayın uzaması (doğal boydan itibaren, M yukarı çıkarsa x > 0).
       Toplam ip uzunluğu sabit olduğundan M x kadar yukarı çıkınca sarkaç
       tarafındaki ip x kadar uzar:  r = l1 + x   (l1: x = 0'daki üst link boyu).
* th1, th2 : düşeyden ölçülen sarkaç açıları.  l2 sabit (rijit alt link).
* Makara kütlesiz ve boyutsuz noktasal kabul edilir, ip esnemez ve gergin
  kalır (gerginlik simulate.py içinde kontrol edilir).

Lagrangian  L = T - V
    T = 1/2 M xd^2 + 1/2 m1 |v1|^2 + 1/2 m2 |v2|^2
    V = 1/2 k x^2 + M g x - m1 g r cos(th1) - m2 g (r cos(th1) + l2 cos(th2))

Çıktı: model_generated.py  (lambdify çıktısının kaynak kodu). Her çalıştırmada
yeniden türetmemek için simulate.py bu dosya varsa onu import eder.
"""
import inspect
import os
import sys
import textwrap

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "model_generated.py")


def derive():
    t = sp.symbols("t")
    M, m1, m2, k, l1, l2, g = sp.symbols("M m1 m2 k l1 l2 g", positive=True)

    x = sp.Function("x")(t)
    th1 = sp.Function("th1")(t)
    th2 = sp.Function("th2")(t)
    q = [x, th1, th2]

    r = l1 + x
    # Makara orijinde, y yukarı
    X1 = r * sp.sin(th1)
    Y1 = -r * sp.cos(th1)
    X2 = X1 + l2 * sp.sin(th2)
    Y2 = Y1 - l2 * sp.cos(th2)

    # --- Kinetik enerji (M düşeyde sadece xd ile hareket eder) ---
    T_M = sp.Rational(1, 2) * M * sp.diff(x, t) ** 2
    T_1 = sp.Rational(1, 2) * m1 * (sp.diff(X1, t) ** 2 + sp.diff(Y1, t) ** 2)
    T_2 = sp.Rational(1, 2) * m2 * (sp.diff(X2, t) ** 2 + sp.diff(Y2, t) ** 2)
    T = T_M + T_1 + T_2

    # --- Potansiyel enerji ---
    V_el = sp.Rational(1, 2) * k * x ** 2           # yay (x = uzama)
    V_g = M * g * x + m1 * g * Y1 + m2 * g * Y2     # M: yükseklik = x (+sabit)
    V = V_el + V_g

    L = T - V

    # --- Euler-Lagrange: d/dt(dL/dqd) - dL/dq = 0 ---
    eqs = [sp.diff(sp.diff(L, sp.diff(qi, t)), t) - sp.diff(L, qi) for qi in q]

    # Dinamik fonksiyonları düz sembollere çevir
    xs, a1s, a2s = sp.symbols("x th1 th2")
    xds, w1s, w2s = sp.symbols("xd th1d th2d")
    xdds, b1s, b2s = sp.symbols("xdd th1dd th2dd")
    rep = {}
    for f, s, sd, sdd in zip(q, (xs, a1s, a2s), (xds, w1s, w2s), (xdds, b1s, b2s)):
        rep[sp.diff(f, t, 2)] = sdd
        rep[sp.diff(f, t)] = sd
        rep[f] = s
    # Önce yüksek türevleri yerine koy (sıra önemli)
    def sub(e):
        e = e.subs({sp.diff(f, t, 2): rep[sp.diff(f, t, 2)] for f in q})
        e = e.subs({sp.diff(f, t): rep[sp.diff(f, t)] for f in q})
        e = e.subs({f: rep[f] for f in q})
        return e

    eqs = [sub(e) for e in eqs]
    qdd = [xdds, b1s, b2s]
    A, b = sp.linear_eq_to_matrix(eqs, qdd)       # A * qdd = b
    A = A.applyfunc(sp.simplify)
    b = b.applyfunc(sp.simplify)

    # qdd = adj(A) b / det(A)  (3x3, kapalı form; sayısal solve gerektirmez)
    detA = sp.simplify(A.det())
    qdd_expr = (A.adjugate() * b).applyfunc(lambda e: e / detA)

    # --- Enerji bileşenleri ve konumlar (düz semboller) ---
    T_s = sub(T)
    Vel_s = sub(V_el)
    Vg_s = sub(V_g)
    pos = sp.Matrix([sub(e) for e in (X1, Y1, X2, Y2)])

    return dict(
        params=(M, m1, m2, k, l1, l2, g),
        q=(xs, a1s, a2s), qd=(xds, w1s, w2s),
        A=A, b=b, qdd=qdd_expr,
        T=sp.simplify(T_s), Vel=Vel_s, Vg=sp.simplify(Vg_s), pos=pos,
        L=L, eqs=eqs,
    )


def _source(name, args, expr):
    f = sp.lambdify(args, expr, modules="numpy", cse=True)
    src = inspect.getsource(f)
    return src.replace("def _lambdifygenerated(", f"def {name}(", 1)


def build(path=OUT_FILE):
    d = derive()
    p = d["params"]
    q, qd = d["q"], d["qd"]
    sig_q = list(q) + list(qd) + list(p)

    parts = []
    parts.append(textwrap.dedent('''\
        # -*- coding: utf-8 -*-
        # OTOMATİK ÜRETİLDİ -- derive.py tarafından yazıldı, elle düzenlemeyin.
        # Parametre sırası: (M, m1, m2, k, l1, l2, g)
        # Durum: q = (x, th1, th2), qd = (xd, th1d, th2d)
        import numpy
        from numpy import sin, cos, array

    '''))
    # ivmeler: (x, th1, th2, xd, th1d, th2d, M, m1, m2, k, l1, l2, g) -> [xdd, th1dd, th2dd]
    parts.append(_source("accel", sig_q, list(d["qdd"])))
    parts.append(_source("mass_matrix", list(q) + list(p), d["A"]))
    parts.append(_source("kinetic", sig_q, d["T"]))
    parts.append(_source("elastic", list(q) + list(p), d["Vel"]))
    parts.append(_source("gravity", list(q) + list(p), d["Vg"]))
    parts.append(_source("positions", list(q) + list(p), list(d["pos"])))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts))
    return d


if __name__ == "__main__":
    d = build()
    print("Lagrangian L =")
    sp.pprint(sp.simplify(d["L"]))
    print("\nKütle matrisi A (A qdd = b):")
    sp.pprint(d["A"])
    print(f"\nYazıldı: {OUT_FILE}")
