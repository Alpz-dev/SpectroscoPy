from matplotlib import pyplot as plt
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
three_color = ["#ffb14e","#ea5f94","#0000ff"]


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
             xticks = None,
             yticks = None,
             linewidth = 1.5,
             linestyle = "solid"):

    plt.figure(fig_num, figsize = (8, 6))

    if scatter:
        plt.scatter(dataset.x, dataset.y, color = color, edgecolor = color, label = dataset.data_label, marker = "^", s = 80)
    else:
        plt.plot(dataset.x, dataset.y, linewidth = linewidth, color = color, label = dataset.data_label, linestyle = linestyle)

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

    if legend:
        plt.legend(loc = "best", fontsize = labelsize - 2, frameon = True, fancybox = False)

    plt.tight_layout()


# Plots multiple Data objects overlapped on a single figure (sep = False) or
# plots multiple Data objects on separate figures (sep = True)
def multi_plot(data_array: np.array,
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
               xticks=None,
               yticks=None,
               linewidth = 1.5,
               linestyle = "solid"):

    if color_scheme is None:
        color_scheme = three_color


    for i in range(data_array.size):

        if sep:
            num = i
        else:
            if fig_num is not None:
                num = fig_num
            else:
                num = 1

        if (isinstance(data_array[i], np.ndarray) or isinstance(data_array[i], list)):
            for elem in data_array[i]:
                uni_plot(elem, num,
                         # color scheme stepped by fixed step if len(dataset) < len(colorscheme)
                         color = color_scheme[((len(color_scheme) // data_array.size) + i) % len(color_scheme)],
                         xlabel = xlabel,
                         ylabel = ylabel,
                         labelsize = labelsize,
                         x_window = x_window,
                         y_window = y_window,
                         legend = legend,
                         scatter = scatter,
                         yticks = yticks,
                         xticks = xticks,
                         linewidth = linewidth,
                         linestyle = linestyle)

        else:
            uni_plot(data_array[i], num,
                     # color scheme stepped by fixed step if len(dataset) < len(colorscheme)
                     color = color_scheme[((len(color_scheme) // data_array.size) + i) % len(color_scheme)],
                     xlabel = xlabel,
                     ylabel = ylabel,
                     labelsize = labelsize,
                     x_window = x_window,
                     y_window = y_window,
                     legend = legend,
                     scatter = scatter,
                     yticks = yticks,
                     xticks = xticks,
                     linewidth = linewidth,
                     linestyle = linestyle)

