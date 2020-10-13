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
    while m // n < 10:
        m = m << 1
    return m


def swap_half_array_between(x: (np.array, np.ndarray, np.nditer)):
    """swap the first half and the second half of array"""
    n = len(x)
    # copy to rewrite
    temp = np.array(x[:n // 2])
    x[:n // 2] = x[n // 2:]
    x[n // 2:] = temp
    return x


def add_zeros(x, m):
    zero = np.zeros(m)
    n = len(x)
    left_index = ((m - n) // 2)
    right_index = n + left_index
    # m = 4, n = 2, left_index = 1
    # m = 8, n = 4, left_index = 2
    zero[left_index:right_index] = x
    return zero


def ft_area_algo(f, n: int, a: float):
    """
    param f: function
    param n: needed vector point count
    param a: the right border of line segment [-a,a]
    return F,b
    Fourier transform and the right border of line segment [-b,b]
    """
    x, h_x = np.linspace(-a, a, n, retstep=True)
    f_val = f(x)
    n = len(f_val)
    m = get_m(n)
    f_val_added_zero = add_zeros(f_val, m)
    f_val_added_swapped = swap_half_array_between(f_val_added_zero)

    F_val_added_swapped = fft(f_val_added_swapped)
    F_val_added_zero = swap_half_array_between(F_val_added_swapped)
    # cut n values
    left_index = ((m - n) // 2) - 1
    right_index = n + left_index
    F_val = F_val_added_zero[left_index: right_index] * h_x

    # checking results
    print(f"""The line segment is [{-a},{a}]
x: \n{x}
n= {n}
m = {m}
f_val is:\n{f_val}\n
The result {F_val}\n
""")
    return x, f_val, F_val, h_x, n, m


def draw_amplitude_and_phase(x, f_val, color: str = "blue", title_note: str = "", xlabel: str = "x", xlim=None):
    """
    param x: abscissas of coordinates
    param f_val: array with complex elements
    param color: color of line
    param title_note: the text note for the title and ordinate axes
    just draw
    and increment the global variable curr_figure
    """
    global curr_figure
    f_val_afin = (np.absolute(f_val), np.angle(f_val, deg=False))

    pylab.figure(curr_figure)
    curr_figure = curr_figure + 1

    axes1 = pylab.subplot(211)
    pylab.xlabel(xlabel)
    pylab.ylabel("Amplitude")
    axes1.set_title("Amplitude of " + title_note)
    pylab.xlim(xlim) if xlim is not None else None
    pylab.plot(x, f_val_afin[0], color=color)

    axes2 = pylab.subplot(212)
    pylab.xlabel(xlabel)
    pylab.ylabel("Angle")
    axes2.set_title("Angle of " + title_note)
    pylab.xlim(xlim) if xlim is not None else None
    pylab.plot(x, f_val_afin[1], color=color)

    print(f"xlim= {xlim}")


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
    x, f_val, F_val, h_x, n, m = ft_area_algo(get_gauss(s), n, a)
    border_x = np.array([-a, a])  # + [-0.5, 0.5]
    gauss_note = f"Gauss,s={s}, a = {a}, n={n}, m={m}"
    ff_gaus = f"fft of Gauss, a = {a} "
    draw_amplitude_and_phase(x, f_val, title_note=gauss_note)
    draw_amplitude_and_phase(x, F_val, title_note=ff_gaus)
    pylab.show()


if __name__ == '__main__':
    # tests()
    main()
