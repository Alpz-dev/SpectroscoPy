
from matplotlib import pyplot as plt
from matplotlib.backend_bases import LocationEvent
import numpy as np

# Color Schemes
cool = ["#31247c", "#0047a5", "#0066be", "#0080c0", "#0099ac", "#00af89", "#00c25c", "#7bd028"]
warm = ["#842013", "#983b15", "#ab5418", "#bc6f1c", "#ca8924", "#d6a52f", "#e0c13e", "#e7de52"]
seashore = ["#1D2F6F", "#8390FA", "#FAC748", "#00BFB2", "#C64191"]
seashore_dark = ["#18265b", "#6a7af9", "#f9bf2f", "#00a69a", "#b73784"]
cyberpunk = ["#E0E31B", "#292837", "#F75C03", "#4554B5", "#B4C1AA"]
skycity = ["#261447", "#FF640A", "#02A9EA", "#DDE000", "#3F403F"]
pastel = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
rainbow = ["#FF6973", "#E8AE5F", "#FFDA4B", "#5FE88C", "#64B8FF"]
black = ["#111133"]
microsoft = ["#fff100", "#ff8c00", "#e81123", "#ec008c", "#68217a", "#00188f", "#00bcf2", "#00b294", "#009e49", "#bad80a"]
true_rainbow = ["#4C0594", "#357ECD", "#24C442", "#E5E33F", "#E57C00", "#CE141F"]
three_color = ["#ea5f94", "#ffb14e","#0000ff"]
three_more_colors=["#7B4B94", "#7D82B8", "#7A9E7E"]


# Plots a single "Data" object
def uni_plot(dataset, fig_num: int,
             color: str = "black",
             xlabel: str = "",
             ylabel: str = "",
             labelsize: int = 20,
             x_window: tuple = None,
             y_window: tuple = None,
             legend = True,
             scatter = False,
             line = False,
             xticks = None,
             yticks = None,
             linewidth = 1.5,
             linestyle = "solid"):

    plt.figure(fig_num, figsize = (8, 6))

    if isinstance(legend, str):
        legend_val = legend
        _legend = True
    else:
        legend_val = dataset.data_label
        _legend = legend

    if scatter:
        plt.scatter(dataset.x, dataset.y, color = color, edgecolor = color, label = legend_val, marker = "^", s = 30)
    elif line:
        plt.vlines(dataset.x, min(dataset.y), dataset.y, linewidth = linewidth, color = color, label = legend_val, linestyle = linestyle)
    else:
        plt.plot(dataset.x, dataset.y, linewidth = linewidth, color = color, label = legend_val, linestyle = linestyle)

    plt.xlabel(xlabel, fontsize = labelsize)
    plt.ylabel(ylabel, fontsize = labelsize)

    plt.yticks(fontsize = labelsize - 4)
    plt.xticks(fontsize = labelsize - 4)

    if xticks is not None:
        plt.xticks(xticks, fontsize = labelsize - 4)
    if yticks is not None:
        plt.yticks(yticks, fontsize = labelsize - 4)

    plt.yticks(fontsize = labelsize - 4)
    plt.xticks(fontsize = labelsize - 4)

    if x_window is not None: plt.xlim(x_window)
    if y_window is not None: plt.ylim(y_window)

    if _legend:
        plt.legend(loc = "best", fontsize = labelsize - 4, frameon = True, fancybox = False)


    plt.minorticks_on()
    plt.tick_params(axis = "x", which = "minor", direction = "in", length = 3, top = True)
    plt.tick_params(axis = "x", which = "major", direction = "in", length = 5, top = True)
    plt.tick_params(axis = "y", which = "minor", direction = "in", length = 3, right = True)
    plt.tick_params(axis = "y", which = "major", direction = "in", length = 5, right = True)
    plt.tight_layout()


# Plots multiple Data objects overlapped on a single figure (sep = False) or
# plots multiple Data objects on separate figures (sep = True)
def multi_plot(data_array: list,
               sep: bool = False,
               color_scheme: list = None,
               xlabel: str = "",
               ylabel: str = "",
               labelsize: int = 20,
               x_window: tuple = None,
               y_window: tuple = None,
               legend = True,
               fig_num = None,
               scatter = False,
               line = False,
               xticks=None,
               yticks=None,
               linewidth = 1.5,
               linestyle = "solid"):

    if color_scheme is None:
        color_scheme = three_color


    for i in range(len(data_array)):

        if sep:
            num = i
        else:
            if fig_num is not None:
                num = fig_num
            else:
                num = 1


        if isinstance(scatter, list):
            _scatter = scatter[i]
        else:
            _scatter = scatter
        
        if isinstance(linestyle, list):
            _linestyle = linestyle[i]
        else:
            _linestyle = linestyle
        
        if isinstance(legend, list):
            _legend = legend[i]
        else:
            _legend = legend

        uni_plot(data_array[i], num,
                # color scheme stepped by fixed step if len(dataset) < len(colorscheme)
                color = color_scheme[i%len(color_scheme)],
                xlabel = xlabel,
                ylabel = ylabel,
                labelsize = labelsize,
                x_window = x_window,
                y_window = y_window,
                legend = _legend,
                scatter = _scatter,
                yticks = yticks,
                xticks = xticks,
                linewidth = linewidth,
                linestyle = _linestyle, 
                line = line)

