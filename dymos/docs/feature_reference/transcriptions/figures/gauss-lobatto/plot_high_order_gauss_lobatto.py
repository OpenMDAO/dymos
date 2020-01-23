import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

p = om.Problem(model=om.Group())
phase = dm.Phase(ode_class=BrachistochroneODE,
                 transcription=dm.GaussLobatto(num_segments=4, order=[3, 5, 3, 5]))
p.model.add_subsystem('phase0', phase)

p.setup()
p['phase0.t_initial'] = 1.0
p['phase0.t_duration'] = 9.0
p.run_model()

grid_data = phase.options['transcription'].grid_data

t_all = p.get_val('phase0.timeseries.time')
t_disc = t_all[grid_data.subset_node_indices['state_disc'], 0]
t_col = t_all[grid_data.subset_node_indices['col'], 0]


def f(x):  # pragma: no cover
    return np.sin(x) / x + 1


def fu(x):  # pragma: no cover
    return (np.cos(x) * x - np.sin(x))/x**2


def plot_01():  # pragma: no cover

    fig, axes = plt.subplots(1, 1)

    ax = axes

    x = np.linspace(1, 10, 100)

    # Plot the segment boundaries
    segends = np.linspace(1, 10, 5)
    for i in range(len(segends)):
        ax.plot((segends[i], segends[i]), (0, 1), linestyle='--', color='gray')
        if i > 0:
            ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
                        arrowprops=dict(arrowstyle='<->'))
            ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
                    ha='center', va='center')

    # Set the bounding box properties
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set the labels
    ax.set_xlabel('time')

    # Remove the axes ticks
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off

    plt.savefig('01_segments.png')


def plot_02():  # pragma: no cover

    fig, axes = plt.subplots(1, 1)

    ax = axes

    x = np.linspace(1, 10, 100)

    # Plot the state time history
    # ax.plot(x, y, 'b-')
    ax.plot(t_all, 0.5*np.ones_like(t_all), 'kx')

    # Plot the segment boundaries
    segends = np.linspace(1, 10, 5)
    for i in range(len(segends)):
        ax.plot((segends[i], segends[i]), (0, 1), linestyle='--', color='gray')
        if i > 0:
            ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
                        arrowprops=dict(arrowstyle='<->'))
            ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
                    ha='center', va='center')

    # Set the bounding box properties
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set the labels
    ax.set_xlabel('time')
    # ax.set_ylabel('state value')

    # Remove the axes ticks
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off

    plt.savefig('02_nodes.png')


def plot_03():  # pragma: no cover

    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        ax = axes[i]

        x = np.linspace(1, 10, 100)
        if i == 0:
            # Plot the state time history
            y = f(x)
            ax.plot(t_disc, f(t_disc), 'bo')
        elif i == 1:
            y = fu(x)
            ax.plot(t_all, fu(t_all), 'rs')

        y_max = np.max(y)
        y_min = np.min(y)

        # Plot the segment boundaries
        segends = np.linspace(1, 10, 5)
        for j in range(len(segends)):
            ax.plot((segends[j], segends[j]), (y_min, y_max), linestyle='--', color='gray',
                    zorder=-100)
            # if i > 0:
            #     # ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
            #     #             arrowprops=dict(arrowstyle='<->'))
            #     # ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
            #     #         ha='center', va='center')

        # Set the bounding box properties
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # print(i)
        # if i==0:
        #     ax.spines['bottom'].set_color('none')

        # Set the labels
        ax.set_xlabel('time')
        if i == 0:
            ax.set_ylabel('state value')
        else:
            ax.set_ylabel('control value')

        # Remove the axes ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

    plt.savefig('03_inputs.png')


def plot_04():  # pragma: no cover

    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        ax = axes[i]

        x = np.linspace(1, 10, 100)
        if i == 0:
            # Plot the state time history
            y = f(x)
            # ax.plot(x, y, 'b-')

            f_all = f(t_all)
            ax.plot(t_all, f_all, 'bo')

        elif i == 1:
            y = fu(x)
            ax.plot(x, y, 'r-')
            ax.plot(t_all, fu(t_all), 'rs')

        y_max = np.max(y)
        y_min = np.min(y)

        # Plot the segment boundaries
        segends = np.linspace(1, 10, 5)
        for j in range(len(segends)):
            ax.plot((segends[j], segends[j]), (y_min, y_max), linestyle='--', color='gray',
                    zorder=-100)

        # Set the bounding box properties
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Set the labels
        ax.set_xlabel('time')
        if i == 0:
            ax.set_ylabel('state value')
        else:
            ax.set_ylabel('control value')

        # Remove the axes ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

    plt.savefig('04_control_rate_interpolation.png')


def plot_05():  # pragma: no cover

    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        ax = axes[i]

        x = np.linspace(1, 10, 100)
        if i == 0:
            # Plot the state time history
            y = f(x)
            f_disc = f(t_disc)
            df_dx_disc = fu(t_disc)
            # ax.plot(x, y, 'b-')
            ax.plot(t_disc, f_disc, 'bo')

            for j in range(len(t_disc)):
                dx = 0.3
                ax.plot((t_disc[j]-dx, t_disc[j]+dx),
                        (f_disc[j]-dx*df_dx_disc[j], f_disc[j]+dx*df_dx_disc[j]), 'r--')

        elif i == 1:
            y = fu(x)
            # ax.plot(x, y, 'r-')
            ax.plot(t_all, fu(t_all), 'rs')
        y_max = np.max(y)
        y_min = np.min(y)

        # Plot the segment boundaries
        segends = np.linspace(1, 10, 5)
        for j in range(len(segends)):
            ax.plot((segends[j], segends[j]), (y_min, y_max), linestyle='--', color='gray',
                    zorder=-100)
            # if i > 0:
            #     # ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
            #     #             arrowprops=dict(arrowstyle='<->'))
            #     # ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
            #     #         ha='center', va='center')

        # Set the bounding box properties
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # print(i)
        # if i==0:
        #     ax.spines['bottom'].set_color('none')

        # Set the labels
        ax.set_xlabel('time')
        if i == 0:
            ax.set_ylabel('state value')
        else:
            ax.set_ylabel('control value')

        # Remove the axes ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

    plt.savefig('05_ode_eval_disc.png')


def plot_06():  # pragma: no cover

    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        ax = axes[i]

        x = np.linspace(1, 10, 100)
        if i == 0:
            # Plot the state time history
            y = f(x)
            ax.plot(x, y, 'b-')

            f_disc = f(t_disc)
            f_all = f(t_all)
            df_dx_disc = fu(t_disc)
            # ax.plot(x, y, 'b-')
            ax.plot(t_all, f_all, 'bo')

            for j in range(len(t_disc)):
                dx = 0.3
                ax.plot((t_disc[j]-dx, t_disc[j]+dx),
                        (f_disc[j]-dx*df_dx_disc[j], f_disc[j]+dx*df_dx_disc[j]), 'r--')

        elif i == 1:
            y = fu(x)
            ax.plot(x, y, 'r-')
            ax.plot(t_all, fu(t_all), 'rs')

        y_max = np.max(y)
        y_min = np.min(y)

        # Plot the segment boundaries
        segends = np.linspace(1, 10, 5)
        for j in range(len(segends)):
            ax.plot((segends[j], segends[j]), (y_min, y_max), linestyle='--', color='gray',
                    zorder=-100)
            # if i > 0:
            #     # ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
            #     #             arrowprops=dict(arrowstyle='<->'))
            #     # ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
            #     #         ha='center', va='center')

        # Set the bounding box properties
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # print(i)
        # if i==0:
        #     ax.spines['bottom'].set_color('none')

        # Set the labels
        ax.set_xlabel('time')
        if i == 0:
            ax.set_ylabel('state value')
        else:
            ax.set_ylabel('control value')

        # Remove the axes ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

    plt.savefig('06_interpolation.png')


def plot_07():  # pragma: no cover

    fig, axes = plt.subplots(2, 1)

    for i in range(2):
        ax = axes[i]

        x = np.linspace(1, 10, 100)
        if i == 0:
            # Plot the state time history
            y = f(x)
            ax.plot(x, y, 'b-')

            f_col = f(t_col)
            f_all = f(t_all)
            df_dx_col_approx = fu(t_col)
            df_dx_col_computed = -0.5 * fu(t_col)
            # ax.plot(x, y, 'b-')
            ax.plot(t_all, f_all, 'bo')

            for j in range(len(t_col)):
                dx = 0.3
                ax.plot((t_col[j]-dx, t_col[j]+dx),
                        (f_col[j]-dx*df_dx_col_approx[j], f_col[j]+dx*df_dx_col_approx[j]), 'r--')
                ax.plot((t_col[j]-dx, t_col[j]+dx),
                        (f_col[j]-dx*df_dx_col_computed[j], f_col[j]+dx*df_dx_col_computed[j]),
                        'k--')

        elif i == 1:
            y = fu(x)
            ax.plot(x, y, 'r-')
            ax.plot(t_all, fu(t_all), 'rs')

        y_max = np.max(y)
        y_min = np.min(y)

        # Plot the segment boundaries
        segends = np.linspace(1, 10, 5)
        for j in range(len(segends)):
            ax.plot((segends[j], segends[j]), (y_min, y_max), linestyle='--', color='gray',
                    zorder=-100)
            # if i > 0:
            #     # ax.annotate('', xy=(segends[i], 0.05), xytext=(segends[i-1], 0.05),
            #     #             arrowprops=dict(arrowstyle='<->'))
            #     # ax.text((segends[i]+segends[i-1])/2, 0.15, 'Segment {0}'.format(i-1),
            #     #         ha='center', va='center')

        # Set the bounding box properties
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # print(i)
        # if i==0:
        #     ax.spines['bottom'].set_color('none')

        # Set the labels
        ax.set_xlabel('time')
        if i == 0:
            ax.set_ylabel('state value')
        else:
            ax.set_ylabel('control value')

        # Remove the axes ticks
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)  # labels along the bottom edge are off

    plt.savefig('07_ode_eval_col.png')


if __name__ == '__main__':  # pragma: no cover
    plot_01()
    plot_02()
    plot_03()
    plot_04()
    plot_05()
    plot_06()
    plot_07()
    plt.show()
