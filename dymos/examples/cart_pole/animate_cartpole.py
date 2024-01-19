import numpy as np


def animate_cartpole(x, theta, force, interval=20, force_scaler=0.1, save_gif=False, gif_fps=20):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import animation
    # x: time history of cart location, 1d vector
    # theta: time history of pole angle, 1d vector
    # force: control input force

    # keep showing the final state for 100 ms
    extend = 500 // interval
    x = np.concatenate((x, np.ones(extend) * x[-1]))
    theta = np.concatenate((theta, np.ones(extend) * theta[-1]))
    force = np.concatenate((force, np.ones(extend) * force[-1]))

    # cart parameters
    cart_width = 0.25
    cart_height = 0.15
    lpole = 0.5

    # path of the pole tip
    x_pole = lpole * np.sin(theta) + x
    y_pole = -lpole * np.cos(theta) + cart_height / 2.0

    # force vector
    arrow_direc = force * force_scaler
    arrow_origin = x - arrow_direc
    arrow_y = cart_height / 2

    # x_lim and y_lim for figure
    xlim = [min(min(x), min(x_pole)) - cart_width / 2 - 0.1, max(max(x), max(x_pole)) + cart_width / 2 + 0.1]
    ylim = [cart_height / 2 - lpole - 0.05, cart_height / 2 + lpole + 0.05]

    fig, ax = plt.subplots()

    def init():
        # initialize figure
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")

    def animate(i):
        ax.clear()
        # plot "floor"
        ax.plot(xlim, [0.0, 0.0], color="k")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # plot cart
        cart = ax.add_patch(
            patches.Rectangle((-cart_width / 2.0 + x[i], 0.0), cart_width, cart_height, edgecolor="C0", linewidth=1, fill=False)
        )
        # plot pole
        pole = ax.plot([x[i], x_pole[i]], [cart_height / 2, y_pole[i]], "o-", lw=2, color="C0")
        # plot pole tip path
        path = ax.plot(x_pole[:i], y_pole[:i], "--", lw=1, color="black")
        # plot force vector
        f = ax.arrow(
            arrow_origin[i],
            arrow_y,
            arrow_direc[i],
            0.0,
            color="C1",
            head_width=0.02,
            head_length=0.03,
            length_includes_head=True,
        )
        return pole, cart, path, f

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=interval, repeat=True)
    if save_gif:
        anim.save("cartpole.gif", dpi=300, writer=animation.PillowWriter(fps=gif_fps))
    plt.show()
