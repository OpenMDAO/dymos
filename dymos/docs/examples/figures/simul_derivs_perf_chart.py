

def simul_derivs_perf_chart():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ('Dense', 'Simul-Derivs', 'Fortran (OTIS4)')
    y_pos = np.arange(len(labels))
    total_time = np.array([38.6422, 2.3723, 1.95])
    deriv_time = np.array([32.898, 1.0266, 0.85])

    # Normalize by dense total time
    total_time /= 38.6422
    deriv_time /= 38.6422

    # data to plot
    n_groups = 3

    # create plot
    fig, ax = plt.subplots()
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(ymin=0.01, ymax=1.0)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, total_time, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Total Time')

    rects2 = plt.bar(index + bar_width, deriv_time, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Sens Time')

    plt.ylabel('Log(Time) (normalized)')
    plt.title('Dymos Performance with and without Sparsity\nCompared to Legacy Optimal Control '
              'Tool OTIS4\nBrachistochrone Problem with 200 3rd Order Gauss-Lobatto Segments')
    plt.xticks(index + bar_width/2, labels)
    plt.legend()

    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    simul_derivs_perf_chart()