import matplotlib.pyplot as plt


def plot_results(axes, title, figsize=(10, 8), p_sol=None, p_sim=None):
    """
    Plot the timeseries results of a Dymos problem using matplotlib.

    Parameters
    ----------
    axes : list of tuple of (str, str, str, str)
        A sequence of tuples wherein each tuple contains ('x path', 'y path' , 'x label', 'y label').
    title : str
        The figure title
    figsize : tuple of (int, int)
        The size of the created figure, in inches
    p_sol : The solution problem instance.
    p_sim : The simulation problem instance.

    Returns
    -------
    fig, axes
        The Figure object and sequence of axes associated with the plot.

    """
    nrows = len(axes)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    fig.suptitle(title)

    if nrows == 1:
        axs = [axs]

    for i, (x, y, xlabel, ylabel) in enumerate(axes):
        axs[i].plot(p_sol.get_val(x),
                    p_sol.get_val(y),
                    marker='o',
                    ms=4,
                    linestyle='None',
                    label='solution' if i == 0 else None)

        axs[i].plot(p_sim.get_val(x),
                    p_sim.get_val(y),
                    marker=None,
                    linestyle='-',
                    label='simulation' if i == 0 else None)

        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        fig.suptitle(title)
        fig.legend(loc='lower center', ncol=2)

    return fig, axs
