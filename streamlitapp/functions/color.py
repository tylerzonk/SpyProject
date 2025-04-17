import seaborn as sns
import matplotlib.pyplot as plt
import random
from matplotlib.collections import PathCollection  # Correct import

def customize_graph(ax):
    # Get the figure object associated with the Axes
    fig = ax.get_figure()
    # Set the figure background color
    fig.patch.set_facecolor('#0F1116') 
    # Set the Axes background color
    ax.set_facecolor('#0F1116')

    # Change the color of the x-axis label, y-axis label, and title to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    # Change the color of the x-axis and y-axis ticks to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # Change the color of the axes spines to white
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Seaborn color palette
    palette = sns.color_palette("Dark2", n_colors=8)  # Dark2 palette with 8 colors
    
    # Randomize the color palette
    # random.shuffle(palette)  # Shuffle the colors to randomize their order
    
    # Find all elements (bars, lines, scatter points, etc.)
    plot_elements = []
    if ax.patches:  # Check for bar plots
        plot_elements = ax.patches
    elif ax.lines:  # Check for line plots
        plot_elements = ax.lines
    elif ax.collections:  # Check for scatter plots (PathCollection elements)
        plot_elements = ax.collections

    # Apply randomized colors from the shuffled palette to the plot elements
    num_elements = len(plot_elements)
    if num_elements > 0:
        for i, element in enumerate(plot_elements):
            # Apply the appropriate color from the palette based on the index
            color = palette[i % len(palette)]  # Cycle through the colors if there are more elements than colors
            if isinstance(element, plt.Rectangle):  # For bar plots (Rectangle elements)
                element.set_facecolor(color)
            elif isinstance(element, plt.Line2D):  # For line plots (Line2D elements)
                element.set_color(color)
            elif isinstance(element, PathCollection):  # For scatter plots (PathCollection elements)
                element.set_edgecolor(color)
                element.set_facecolor(color)