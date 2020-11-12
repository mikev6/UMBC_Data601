import matplotlib.pyplot as plt
def draw_line(X,
              y,
              intercept=7,
              slope=0.04,
              figsize=(15, 5),
              xlabel='Tv Advertisements',
              ylabel='Sales',
              title='A prediction for Sales'):
    """
    draws a line with given intercept and slope together with given data.
    parameters:
    X: array
    y: array
    intercept: float, preferably between 5 and 9 in this case
    slope: float, preferably between 0.02 and 0.08
    xlabel: str, label of the x-axis in the figure.
    ylabel: str, label of the y-axis in the figure.
    return: a figure with data and a regression line with given intercept and slope.
    """
    # find the predicted values. These points lie on the line with
    # given slope and intecept
    y_pred = intercept + slope * X

    # create a new figure and set the figure size
    plt.figure(figsize= figsize)

    # plot data points as scatter
    plt.scatter(x=X, y=y)

    # plot the prediction line
    plt.plot(X, y_pred, c='r', label='Regression Line')

    # set labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # set the title of the figure
    plt.title(title)

    plt.legend()
    plt.show()
    return

