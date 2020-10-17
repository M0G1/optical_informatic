import numpy as np
from scipy.fft import fft
from matplotlib import pylab

curr_figure = 0


def get_gauss(s: (int, float, complex)):
    return lambda x: np.exp(-np.dot(s, x ** 2))


def get_m(n: int):
    # получаем номер старшего не знакового бита равного единице
    elder_bit_num = int.bit_length(n) + 1
    m = 1 << elder_bit_num
    # делаем M намного больше n
    while m // n < 4:
        m = m << 1
    return m


def swap_half_array_between(x: (np.array, np.ndarray, np.nditer)):
    """swap the first half and the second half of array"""
    n = len(x)
    ans = np.ndarray(shape=(n,))
    # copy to rewrite
    ans[:n // 2] = np.array(x[n // 2:])
    ans[n // 2:] = np.array(x[:n // 2])
    return ans


def add_zeros(x, m):
    zero = np.zeros(m)
    n = len(x)
    left_index = ((m - n) // 2)
    right_index = n + left_index
    # m = 4, n = 2, left_index = 1
    # m = 8, n = 4, left_index = 2
    zero[left_index:right_index] = x.copy()
    return zero


def ft_area_algo(f, n: int, a: float, ft):
    """
    param f: function
    param n: needed vector point count
    param a: the right border of line segment [-a,a]
    param ft: Fourier transform
    return F,b
    Fourier transform and the right border of line segment [-b,b]
    """
    x, h_x = np.linspace(-a, a, n, retstep=True)
    f_val = f(x)
    n = len(f_val)
    m = get_m(n)
    f_val_added_zero = add_zeros(f_val, m)
    f_val_added_swapped = swap_half_array_between(f_val_added_zero)

    F_val_added_swapped = ft(f_val_added_swapped)
    F_val_added_zero = swap_half_array_between(F_val_added_swapped)
    # cut n values
    left_index = (m - n) // 2
    # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
    right_index = n + left_index
    F_val = F_val_added_zero[left_index: right_index] * h_x
    get_b = lambda n, a: (n / (2 * a)) ** 2

    # checking results
    print(f"""The line segment is [{-a},{a}]
n= {n}
m = {m}
h_x = {h_x}
x: \n{x}\n
f_val is:\n{f_val}\n
The result {F_val}\n
""")
    return x, f_val, F_val, h_x, m, get_b(n, a)


def draw_amplitude_and_phase(x, f_val, colors: str = "blue", title_note: str = "", xlabel: str = "x", xlim=None,
                             labels: list = None):
    """
    param x: abscissas of coordinates
    param f_val: array with complex elements
    param color: color of line
    param title_note: the text note for the title and ordinate axes
    just draw
    and increment the global variable curr_figure
    """
    global curr_figure

    x_list, f_val_list = [], []
    if labels is None:
        labels = [""] * len(x)
    if not isinstance(colors, list):
        colors = ("blue", "red", "green", "black", "purple", "pink")
    if not isinstance(x, list):
        x_list = [x]
        f_val_afin = (np.absolute(f_val), np.angle(f_val, deg=False))
        f_val_list = [f_val_afin]
    else:
        x_list = x
        for i in range(len(f_val)):
            f_val_list.append((np.absolute(f_val[i]), np.angle(f_val[i], deg=False)))

    pylab.figure(curr_figure)
    curr_figure = curr_figure + 1

    axes1 = pylab.subplot(211)
    pylab.xlabel(xlabel)
    pylab.ylabel("Amplitude")
    axes1.set_title("Amplitude of " + title_note)
    pylab.xlim(xlim) if xlim is not None else None
    for i in range(len(x_list)):
        pylab.plot(x_list[i], f_val_list[i][0], color=colors[i], label=labels[i])

    axes2 = pylab.subplot(212)
    pylab.xlabel(xlabel)
    pylab.ylabel("Angle")
    axes2.set_title("Angle of " + title_note)
    pylab.xlim(xlim) if xlim is not None else None
    for i in range(len(x_list)):
        pylab.plot(x_list[i], f_val_list[i][1], color=colors[i], label=labels[i])


def left_triangle_Fourier(f_val):
    arg = None
    N = len(f_val)
    xx = np.linspace(0, N, N)
    xx, ksi_ksi = np.meshgrid(xx, xx)
    core_val = np.exp(-2 * np.pi * 1j * xx * ksi_ksi / N)
    # print(f"core_val \n{core_val}\nshpape {core_val.shape}")
    temp = f_val[len(f_val) - 1]
    f_val[len(f_val) - 1] = 0
    # left triangle method integrating
    F_val = np.matmul(core_val, f_val)
    f_val[len(f_val) - 1] = temp
    return F_val


def left_triangle_Fourier_arg(f_val, x, ksi):
    arg = None
    N = len(f_val)
    xx, ksi_ksi = np.meshgrid(x, ksi)
    core_val = np.exp(-2 * np.pi * 1j * xx * ksi_ksi)
    # print(f"core_val \n{core_val}\nshpape {core_val.shape}")

    # left triangle method integrating
    F_val = np.matmul(core_val, f_val)
    return F_val


def tests():
    # # swap test
    # ar = np.array([1, 2, 3, 4])
    # ar_swaped = swap_half_array_between(ar)
    # print(f"before:\n{ar}\nafter:\n{ar_swaped}")

    # m = 8
    # x = [1, 2, 3, 4]
    # x_res = add_zeros(x, m)
    pass


def main():
    n = 100
    a = 5
    s = 1
    x, f_val, F_val, h_x, m, b = ft_area_algo(get_gauss(s), n, a, fft)
    x_for_b = np.linspace(-b, b, n)
    print(f"fft: h_x={h_x}, m={m}, b={b}")

    xx, ff_val, F_tri_val, hh_x, mm, bb = ft_area_algo(get_gauss(s), n, a, left_triangle_Fourier)
    # F_tri_val_arg = left_triangle_Fourier_arg(f_val, x, x_for_b)
    print(f"fft: hh_x={hh_x}, m={mm}, b={bb}")
    print(
        f"""difference in values
    x:{np.max(np.abs(x - xx))}
    f_val:{np.max(np.abs(f_val - ff_val))}
    F_val:{np.max(np.abs(F_tri_val - F_val))}
    h_x:{np.max(np.abs(h_x - hh_x))}
    m:{np.max(np.abs(m - mm))}
    b:{np.max(np.abs(b - bb))}
""")

    border_x = np.array([-a, a])  # + [-0.5, 0.5]
    gauss_note = f"Gauss, s={s}"
    b_note = f", b = {b}"
    ff_gaus = "fft of Gauss" + b_note
    amn = f", a = {a}, n={n}, m={m}"
    union_note = gauss_note + " and " + ff_gaus + amn
    labels = ["left triangle Gauss", " fft Gauss", " Gauss"]
    left_trian_Fourier_note = "Fourier of Gauss left triangle" + b_note

    draw_amplitude_and_phase([x_for_b, x_for_b, x], [F_tri_val, F_val, f_val],
                             title_note=union_note, labels=labels)
    draw_amplitude_and_phase(x, f_val, title_note=gauss_note + amn)
    # draw_amplitude_and_phase(x_for_b, F_val, title_note=ff_gaus + amn)
    # draw_amplitude_and_phase(x_for_b, F_tri_val, title_note=left_trian_Fourier_note)
    # pylab.legend()
    pylab.show()


if __name__ == '__main__':
    # tests()
    main()
