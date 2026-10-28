# -*- coding: utf-8 -*-
# OTOMATİK ÜRETİLDİ -- derive.py tarafından yazıldı, elle düzenlemeyin.
# Parametre sırası: (M, m1, m2, k, l1, l2, g)
# Durum: q = (x, th1, th2), qd = (xd, th1d, th2d)
import numpy
from numpy import sin, cos, array



def accel(x, th1, th2, xd, th1d, th2d, M, m1, m2, k, l1, l2, g):
    x0 = th1 - th2
    x1 = cos(x0)
    x2 = l2**2
    x3 = m2**2
    x4 = x2*x3
    x5 = x1*x4
    x6 = sin(x0)
    x7 = l1*x6
    x8 = x*x6
    x9 = x5*x7 + x5*x8
    x10 = sin(th1)
    x11 = g*x10
    x12 = l1*m1
    x13 = m2*x11
    x14 = m1*x
    x15 = th1d*xd
    x16 = 2*x15
    x17 = 2*l1
    x18 = l2*m2
    x19 = th2d**2*x18
    x20 = -l1*x13 - m2*x*x16 - m2*x15*x17 - x*x13 - x11*x12 - x11*x14 - x12*x16 - x14*x16 - x19*x7 - x19*x8
    x21 = l1**2
    x22 = m1*m2
    x23 = l2*x6
    x24 = x22*x23
    x25 = x**2
    x26 = x*x17
    x27 = x21*x3
    x28 = x25*x3
    x29 = x*x3
    x30 = x17*x29
    x31 = -x21*x24 - x23*x27 - x23*x28 - x23*x30 - x24*x25 - x24*x26
    x32 = sin(th2)
    x33 = 2*x1
    x34 = th1d**2
    x35 = x18*(-g*x32 + l1*x34*x6 + x*x34*x6 - x15*x33)
    x36 = g*cos(th1)
    x37 = l1*x34
    x38 = x*x34
    x39 = -M*g - k*x + m1*x36 + m1*x37 + m1*x38 + m2*x36 + m2*x37 + m2*x38 + x1*x19
    x40 = x2*x22
    x41 = x21*x4
    x42 = x25*x4
    x43 = x26*x4
    x44 = x1**2
    x45 = m1**2
    x46 = M*m1
    x47 = M*m2
    x48 = 1/(m2*x2*(l1 + x)**2*(x10**2*x47 - x10*x32*x33*x47 + x22 + x32**2*x47 + x45 + x46))
    x49 = l2*x1
    x50 = l1*x49
    x51 = x*x49
    x52 = -x22*x50 - x22*x51 - x29*x49 - x3*x50 - x47*x50 - x47*x51
    x53 = 2*x22
    return [x48*(x20*x9 + x31*x35 + x39*(x21*x40 + x25*x40 + x26*x40 - x41*x44 + x41 - x42*x44 + x42 - x43*x44 + x43)), x48*(x20*(x2*x47 - x4*x6**2 + x4 + x40) + x35*x52 + x39*x9), x48*(x20*x52 + x31*x39 + x35*(4*l1*x*x22 + x21*x45 + x21*x46 + x21*x47 + x21*x53 + x25*x45 + x25*x46 + x25*x47 + x25*x53 + x26*x45 + x26*x46 + x26*x47 + x27 + x28 + x30))]


def mass_matrix(x, th1, th2, M, m1, m2, k, l1, l2, g):
    x0 = m1 + m2
    x1 = th1 - th2
    x2 = l2*m2
    x3 = x2*sin(x1)
    x4 = l1 + x
    x5 = x2*x4*cos(x1)
    return array([[M + x0, 0, x3], [0, x0*x4**2, x5], [x3, x5, l2**2*m2]])


def kinetic(x, th1, th2, xd, th1d, th2d, M, m1, m2, k, l1, l2, g):
    x0 = xd**2
    x1 = th1d**2
    x2 = 2*x
    x3 = l1**2*x1 + l1*x1*x2 + x**2*x1 + x0
    x4 = th1 - th2
    x5 = l2*th2d
    x6 = 2*x5
    x7 = th1d*cos(x4)
    return (1/2)*M*x0 + (1/2)*m1*x3 + (1/2)*m2*(l1*x6*x7 + l2**2*th2d**2 + x2*x5*x7 + x3 + x6*xd*sin(x4))


def elastic(x, th1, th2, M, m1, m2, k, l1, l2, g):
    return (1/2)*k*x**2


def gravity(x, th1, th2, M, m1, m2, k, l1, l2, g):
    x0 = (l1 + x)*cos(th1)
    return g*(M*x - m1*x0 - m2*(l2*cos(th2) + x0))


def positions(x, th1, th2, M, m1, m2, k, l1, l2, g):
    x0 = l1 + x
    x1 = x0*sin(th1)
    x2 = cos(th1)
    x3 = -x0
    return [x1, x2*x3, l2*sin(th2) + x1, -l2*cos(th2) + x2*x3]
