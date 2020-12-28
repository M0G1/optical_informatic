import numpy as np
from scipy import integrate
from scipy.special import *
from matplotlib import pylab
import mpmath
import time
import lab2


def create_radial_symmetric_image(f_val, m: int):
    n = len(f_val)
    arr_2d = np.zeros((2 * n, 2 * n), dtype=np.complex)
    xs, ys = np.meshgrid(np.arange(0, 2 * n), np.arange(0, 2 * n))
    xs = xs - n
    ys = ys - n
    dist = np.round(np.sqrt(xs ** 2 + ys ** 2)).astype(np.int)
    mask = dist < n
    arr_2d[mask] = f_val[dist[mask]]
    fi = np.arctan2(ys, xs)
    return arr_2d * np.exp(complex(0, 1) * m * fi)


def hankel_transform(f_val, p_val, R: float, m: int):
    """
        Hankel transform
        f_val - function values
        p_val = ρ_val - transform result domain
        R - upper limit of integral
        m - power of Bessel
    """
    n = len(f_val)
    new_x = p_val
    Y = np.zeros(n, dtype=np.complex128)
    for i, j in zip(new_x, range(n)):
        Y[j] = np.sum(f_val * jv(m, 2 * np.pi * p_val * i) * p_val * (R / n))
    return Y * (2 * np.pi / (complex(0, 1) ** m))


def func(r, p: int, n: int):
    return np.exp(-r ** 2) * (r ** np.abs(p)) * lagerr(np.abs(p), n, r ** 2)


def lagerr(p, n, r):
    L = 0
    for j in range(0, n + 1):
        L += ((-1) ** j) * coef(n - j, n + p) * ((r ** j) / np.math.factorial(j))
    return L


def coef(k, a):
    C = 1
    for j in range(0, k):
        C *= a - j
    return C / np.math.factorial(k)


def im_tit(s: str):
    """
    return a tuple with title notes for amplitude and phase
    """
    return "Amplitude " + s, "Phase " + s


def main():
    R = 5
    count = 512
    n = 1
    m = -3
    M = 1 << (int.bit_length(count) + 3)
    M = 16*count
    p = 3
    x, h_x = np.linspace(0, R, count, retstep=True)
    f_val = func(x, p, n)
    rad_sym_mat = create_radial_symmetric_image(f_val, m)

    lab2.draw_amplitude_and_phase(x, f_val, title_note="function", is_need_grid=True)
    lab2.draw_amplitude_and_phase_image(rad_sym_mat, im_tit("restored image"))

    h_start = time.time()
    F_val_Hank = hankel_transform(f_val, x, R, m)
    h_end = time.time()

    rad_sym_Hankel_transform = create_radial_symmetric_image(F_val_Hank, m)

    print("Working time of Hankel transform ", (h_end - h_start))
    lab2.draw_amplitude_and_phase(x, F_val_Hank, title_note="after Hankel transform", is_need_grid=True)
    lab2.draw_amplitude_and_phase_image(rad_sym_Hankel_transform, im_tit("restored image after Hankel transform"))

    f_start = time.time()
    matrix_Fourier_transform, inner_m, new_border = lab2.ft_finite_algo_2d(rad_sym_mat, R, h_x, m=M)
    f_stop = time.time()

    # print("shape ", matrix_Fourier_transform.shape, "\ndtype", matrix_Fourier_transform.dtype, "\n")
    print(f"m = {inner_m}, b = {new_border}, n = {count}")
    print("Working time of Fourier transform", (f_stop - f_start))
    lab2.draw_amplitude_and_phase_image(matrix_Fourier_transform,
                                        im_tit(f"restored image after Fourier transform"))
    #
    # difference = rad_sym_Hankel_transform - matrix_Fourier_transform
    # lab2.draw_amplitude_and_phase_image(difference, "difference in values of transforms")

    pylab.show()


if __name__ == '__main__':
    main()
