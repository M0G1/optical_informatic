import numpy as np

curr_figure = 0


def gaus(s, x):
    return np.exp(-np.dot(s, x ** 2))


def get_m(n: int):
    # получаем номер старшего не знакового бита равного единице
    elder_bit_num = int.bit_length(n) + 1
    m = 1 << elder_bit_num
    # делаем M намного больше n
    while m // n < 10:
        m = m << 1
    return m


def add_zeros(x, m):
    zero = np.zeros(m)
    n = len(x)
    left_index = ((m - n) // 2) - 1
    right_index = n + left_index
    # m = 4, n = 2, left_index = 1
    zero[left_index:right_index] = x
    return x

def ft_area_algo(f, n: int, a: float):
    """
    param f: function
    param n: needed vector point count
    param a: the right border of line segment [-a,a]
    return F,b
    Fourier transform and the right border of line segment [-b,b]
    """
    x = np.linspace(-a, a, n)
    f_val = f(x)


def main():
    pass


if __name__ == '__main__':
    main()
