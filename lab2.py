import numpy as np
from scipy.fft import fft
from matplotlib import pylab
from scipy import integrate

curr_figure = 0


def get_gauss(s: (int, float, complex)):
    return lambda x: np.exp(-np.dot(s, x ** 2))


def get_gauss_2d(s: (int, float, complex), p: (int, float, complex)):
    return lambda x, y: np.exp(-(s * x ** 2 + p * y ** 2))


def get_m(n: int):
    # получаем номер старшего не знакового бита равного единице
    elder_bit_num = int.bit_length(n) + 1
    m = 1 << elder_bit_num
    # делаем M намного больше n
    while m // n < 2:
        m = m << 1
    return m


def swap_half_array_between(x: (np.array, np.ndarray, np.nditer)):
    """swap the first half and the second half of array"""
    n = len(x)
    ans = np.ndarray(shape=(n,), dtype=complex)
    # copy to rewrite
    ans[:n // 2] = np.array(x[n // 2:])
    ans[n // 2:] = np.array(x[:n // 2])
    return ans


def add_zeros(x, m):
    zero = np.zeros(m, dtype=np.complex_)
    n = len(x)
    left_index = ((m - n) // 2)
    right_index = n + left_index
    # m = 4, n = 2, left_index = 1
    # m = 8, n = 4, left_index = 2
    zero[left_index:right_index] = x.copy()
    return zero


def ft_finite_algo(f_val, a: float, h_x: float, m: int = None):
    """
    param f_val: function values
    param a: right border of integrate interval
    param m:  add zeros in both sides before length
    return F,b
    Fourier transform and the right border of line segment [-b,b]
    """
    n = len(f_val)
    if not m:
        m = get_m(n)
    f_val_added_zero = add_zeros(f_val, m)
    f_val_added_swapped = swap_half_array_between(f_val_added_zero)
    F_val_added_swapped = np.fft.fft(f_val_added_swapped)
    F_val_added_zero = swap_half_array_between(F_val_added_swapped)
    # cut n values
    left_index = (m - n) // 2
    # 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
    right_index = n + left_index
    F_val = F_val_added_zero[left_index: right_index] * h_x
    get_b = lambda n, a: (n ** 2 / (4 * a * m))

    #     print(f"""The line segment is [{-a},{a}]
    # n= {n}
    # m = {m}
    # h_x = {h_x}
    # f_val is:\n{f_val}\n
    # The result {F_val}\n
    # """)
    return F_val, m, get_b(n, a)


def ft_finite_algo_2d(matrix, a: float, h_x: float, m: int = None):
    n = matrix.shape[0]
    is_m_given = m is not None
    b = None
    answer = np.zeros(matrix.shape, dtype=matrix.dtype)
    for i in range(n):
        if is_m_given:
            answer[i, :], m, b = ft_finite_algo(matrix[i, :], a, h_x, m=m)
        else:
            answer[i, :], m, b = ft_finite_algo(matrix[i, :], a, h_x)

    for j in range(n):
        if is_m_given:
            answer[:, j], m, b = ft_finite_algo(answer[:, j], a, h_x, m=m)
        else:
            answer[:, j], m, b = ft_finite_algo(answer[:, j], a, h_x)

    return answer, m, b


def ft_finite_num(f, a: float, n: int):
    m = get_m(n)
    b = (n ** 2 / (4 * a * m))
    h_b = 2 * b / (n - 1)
    y = np.zeros(n, dtype=np.complex_)
    for i in range(n):
        u = -b + i * h_b
        integrate_func = lambda x: f(x) * np.exp(-2 * np.pi * u * x * complex(0, 1))
        y[i] = integrate.quad(integrate_func, -a, a)[0]
        del integrate_func
    return y, m, b


def ft_finite_num_2d(f, a: float, n: int):
    # Doesn't need. I am silly and done it. And it works very slowly.
    m, b = None, None
    matrix = np.zeros((n, n), dtype=np.complex_)
    # из-за разделимости функции двойной интеграл можно переписать
    # через произведение интегралов по отдельным переменным, то есть через 1D преобразования
    for i in range(n):
        matrix[i], m, b = ft_finite_num(f, a, n)
        print(i)
    print()
    for j in range(n):
        ans = ft_finite_num(f, a, n)
        print(f"j = {j}")
        matrix[:, j], m, b = matrix[:, j] * ans[0], ans[1], ans[2]

    return matrix, m, b


def draw_amplitude_and_phase(x, f_val, colors: str = "blue", title_note: str = "", xlabel: str = "x", xlim=None,
                             labels: list = None, alpha: float = 1, is_vertical=True, is_need_grid: bool = False):
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
    is_legend_on = True
    if labels is None:
        is_legend_on = False
        labels = [""] * len(x)
    if not isinstance(colors, list):
        colors = ("blue", "red", "green", "pink", "yellow", "purple", "black")
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

    fig, axes = None, None
    if is_vertical:
        fig, axes = pylab.subplots(2, 1)
    else:
        fig, axes = pylab.subplots(1, 2)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Amplitude of " + title_note)
    axes[0].xlim(xlim) if xlim is not None else None
    for i in range(len(x_list)):
        axes[0].plot(x_list[i], f_val_list[i][0], color=colors[i], label=labels[i], alpha=alpha)
    axes[0].legend() if is_legend_on else None
    if is_need_grid:
        axes[0].grid(True)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Phase")
    axes[1].set_title("Phase of " + title_note)
    axes[1].xlim(xlim) if xlim is not None else None
    for i in range(len(x_list)):
        axes[1].plot(x_list[i], f_val_list[i][1], color=colors[i], label=labels[i], alpha=alpha)
    if is_need_grid:
        axes[1].grid(True)


def draw_amplitude_and_phase_image(image, titles=None, vmin=None, vmax=None):
    global curr_figure
    pylab.figure(curr_figure)
    curr_figure = curr_figure + 1
    print(f"cur_figure {curr_figure}")

    # print(f"{}")
    fig, axes = pylab.subplots(1, 2, figsize=(12, 5))
    amplitude = axes[0].imshow(np.absolute(image), cmap="hot")
    fig.colorbar(amplitude, ax=axes[0])
    phase = axes[1].imshow(np.angle(image), cmap="hot", vmin=vmin, vmax=vmax)
    fig.colorbar(phase, ax=axes[1])
    if titles is not None:
        for i in range(2):
            axes[i].set_title(titles[i])


def draw_3d_surface(x, y, z, title=None):
    global curr_figure
    curr_figure = curr_figure + 1
    figure = pylab.figure(curr_figure)
    print(f"cur_figure {curr_figure} 3dd")

    ax = figure.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z)
    ax.set_title(title) if title is not None else None


def tri(x):
    abs_x = np.abs(x)
    return np.where(abs_x < 1, 1 - abs_x, 0)


def tri2d(x, y):
    # https://www.sciencedirect.com/science/article/pii/S0895717711007266
    return tri(x) * tri(y)


def gauss_fft_script():
    n = 150
    a = 5
    s = 1
    x, h_x = np.linspace(-a, a, n, retstep=True)
    gauss = get_gauss(s)
    y = gauss(x)
    F_val, m, b = ft_finite_algo(y, a, h_x)
    x_for_b = np.linspace(-b, b, n)
    print(f"fft: h_x={h_x}, m={m}, b={b}")

    F_num_val, mm, bb = ft_finite_num(gauss, a, n)
    # F_tri_val_arg = left_triangle_Fourier_arg(f_val, x, x_for_b)
    print(f"fft: m={mm}, b={bb}")
    print(
        f"""difference in values
        F_val:{np.max(np.abs(F_num_val - F_val))}
        m:{np.max(np.abs(m - mm))}
        b:{np.max(np.abs(b - bb))}
    """)

    border_x = np.array([-a, a])  # + [-0.5, 0.5]
    gauss_note = f"Gauss, s={s}"
    b_note = ", b = %.2f" % b
    ff_gaus = "fft of Gauss" + b_note
    amn = f",\n a = {a}, n={n}, m={m}"
    union_note = gauss_note + " and " + ff_gaus + amn
    labels = ["quad integrate Ft", " fft of Gauss"]

    draw_amplitude_and_phase([x_for_b, x_for_b], [F_num_val, F_val],
                             title_note=union_note, labels=labels)
    draw_amplitude_and_phase(x, y, title_note=gauss_note + f" a = {a}")
    # draw_amplitude_and_phase(x_for_b, F_val, title_note=ff_gaus + amn)
    # draw_amplitude_and_phase(x_for_b, F_num_val, title_note=left_trian_Fourier_note)
    # pylab.legend()


def gauss_2d_fft_script():
    n = 100
    a = 4
    s = 1
    p = 1
    x, h_x = np.linspace(-a, a, retstep=True)
    gauss_2d = get_gauss_2d(s, p)
    xx, yy = np.meshgrid(x, x)
    z = gauss_2d(xx, yy)
    F_val, m, b = ft_finite_algo_2d(z, a, h_x)
    # F_num_val, m, b = ft_finite_num_2d(get_gauss(1), a, n)

    # print(F_val.shape)
    # print(F_num_val.shape)

    draw_amplitude_and_phase_image(z, [f"Amplitude of Gauss s = {s}", f"Phase of Gauss p = {p}"])
    draw_amplitude_and_phase_image(F_val, ["Amplitude of FFT", "Phase of FFT"])

    # draw_amplitude_and_phase_image(F_num_val)


def tri_fft_script():
    n = 100
    a = 4
    x, h_x = np.linspace(-a, a, n, retstep=True)
    y = tri(x)
    F_val, m, b = ft_finite_algo(y, a, h_x)
    x_for_b = np.linspace(-b, b, n)
    analicit = np.sinc(x_for_b) ** 2

    F_num_val, mm, bb = ft_finite_num(tri, a, n)

    labels = [
        "numeric",
        "algo",
        "analitic"
    ]
    xlim = None  # [-2.1,-1.9]

    draw_amplitude_and_phase(x, y, title_note="triangular function a=%.2f" % a)
    draw_amplitude_and_phase([x_for_b, x_for_b, x_for_b], [F_num_val, F_val, analicit],
                             title_note="fft and ft of tri b=%.2f, a=%.2f\nn=%d, m = %d" % (b, a, n, m), xlim=xlim,
                             labels=labels)


def tri_fft_script_2d():
    n = 100
    a = 2
    x, h_x = np.linspace(-a, a, n, retstep=True)
    xx, yy = np.meshgrid(x, x)
    z = tri2d(xx, yy)
    F_val, m, b = ft_finite_algo_2d(z, a, h_x)

    x_b = np.linspace(-b, b, n)
    xx_b, yy_b = np.meshgrid(x_b, x_b)
    analitic = (np.sinc(xx_b) * np.sinc(yy_b)) ** 2
    print(F_val.shape)
    print(f"m = {m}")
    print(z.shape)
    print(np.max(np.max(z)))

    draw_amplitude_and_phase_image(z, [f"amplitude of tri(x,y), a = {a}", "phase of tri(x,y)"])
    draw_amplitude_and_phase_image(F_val, [f"amplitude of fft, a = {a}", f"phase of fft, b = {b}"])
    draw_amplitude_and_phase_image(analitic, [f"amplitude of sinc^2(x,y), a = {a}", f"phase of sinc^2(x,y), b = {b}"])
    draw_3d_surface(xx_b, yy_b, F_val, "algo")
    draw_3d_surface(xx_b, yy_b, analitic, "analitic")
    draw_3d_surface(xx, yy, z, "tri(x,y)")


def main():
    gauss_fft_script()
    # tri_fft_script()
    # gauss_2d_fft_script()
    # tri_fft_script_2d()
    pylab.show()


def tests():
    global curr_figure
    # # swap test
    # ar = np.array([1, 2, 3, 4])
    # ar_swaped = swap_half_array_between(ar)
    # print(f"before:\n{ar}\nafter:\n{ar_swaped}")

    # m = 8
    # x = [1, 2, 3, 4]
    # x_res = add_zeros(x, m)

    x = np.linspace(-2, 2, 100)
    # tri_val = tri(x)
    # print(x)
    # print(tri_val)
    xx, yy = np.meshgrid(x, x)
    z = tri2d(xx, yy)

    draw_3d_surface(xx, yy, z)
    pylab.show()

    pass


if __name__ == '__main__':
    # tests()
    main()
