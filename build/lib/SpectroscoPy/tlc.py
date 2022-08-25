from SpectroscoPy.data import *
import SpectroscoPy.plot as plot
from PIL import Image
from PIL import ImageOps
import numpy as np
# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/CyPT Cropped4.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))



# test = Data(np.array(x), np.array(y)).normalize().flip().translate(dy = -0.191)
# test.data_label = "Ni$_{4}$(CyPT)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [170, 162, 150], x_window=(176, 140))
# #print(test_deconv)
# plot.multi_plot([test.range((0, 1))] + 
# [test_deconv.bands_data[i].range((0, 1)) 
# for i in range(len(test_deconv.bands_data))] + [test_deconv.fit_data.range((0, 1))], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4] + ["red"], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "dashed", "dashed", "dashed", "solid"], scatter = [False, False, False, False, False], legend = ["Ni$_{n}$(CyPT)$_{2n}$", "Ni$_{6}$(CyPT)$_{12}$", "Ni$_{5}$(CyPT)$_{10}$", "Ni$_{4}$(CyPT)$_{8}$", "Fit"])
# plt.yticks([])
# plot.uni_plot(test_deconv.residual_data.range((0, 1)), 2, x_window = (0, 1), scatter = True)



# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# area_1 = 0
# for i in range(0, 4):
#     band = test_deconv.bands[i]
#     area_1 += band.area()

# area_2 = test_deconv.bands[4].area()
# area_3 = test_deconv.bands[5].area()

# total_area = test.area(test.find_closest(25), test.find_closest(200))
# print(area_1, area_1/total_area)
# print(area_2, area_2/total_area)
# print(area_3, area_3/total_area)

# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/PET TLC Cropped.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))



# test = Data(np.array(x), np.array(y)).normalize().translate(dy = -0.03162)
# uni_plot(test, 30)
# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [200-144, 200-127, 200 - 107], x_window=(200 - 150, 200 - 93), save_name = "vectorized test2")
# #print(test_deconv)
# #   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1))] + 
# [test_deconv.bands_data[i].range((0, 1)) 
# for i in range(len(test_deconv.bands_data))], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "solid", "dashed", "dashed", "dashed"], scatter = [False, False, False, False, False], legend = ["Ni$_{n}$(PET)$_{2n}$", "TEST", "Ni$_{6}$(PET)$_{12}$", "Ni$_{5}$(PET)$_{10}$", "Ni$_{4}$(PET)$_{8}$"])
# plt.yticks([])
# plot.uni_plot(test_deconv.residual_data.range((0, 1)), 2, x_window = (0, 1), scatter = True)



# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())


# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/CyHT Cropped.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# std = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))
#     std.append(np.std(data[i]))



# test = Data(np.array(x), np.array(y)).normalize()
# test.stds = np.array(std)
# uni_plot(test, 30)

# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [37.1, 64.2, 78.2], x_window=(28, 95))
# #print(test_deconv)
# #   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1)).flip()] + 
# [test_deconv.bands_data[i].range((0, 1)).flip() 
# for i in range(len(test_deconv.bands_data))] + [test_deconv.fit_data.range((0,1)).flip()], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "dashed", "dashed", "dashed", "dashed"], scatter = [False, False, False, False, False], legend = ["Ni$_{n}$(CyHT)$_{2n}$", "Ni$_{6}$(CyHT)$_{12}$", "Ni$_{5}$(CyHT)$_{10}$", "Ni$_{4}$(CyHT)$_{8}$", "Fit"])
# plt.yticks([])
# plot.plt.tight_layout()
# plot.uni_plot(test_deconv.residual_data.flip(), 2, x_window = (0, 1), scatter = False)



# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# plot.plt.show()

# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/PET TLC Cropped.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))



# test = Data(np.array(x), np.array(y)).normalize().translate(dy = -0.03162)
# uni_plot(test, 30)
# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [200-144, 200-127, 200 - 107], x_window=(200 - 150, 200 - 93), save_name = "vectorized test2")
# #print(test_deconv)
# #   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1)).flip()] + 
# [test_deconv.bands_data[i].range((0, 1)).flip() 
# for i in range(len(test_deconv.bands_data))], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "dashed", "dashed", "dashed"], scatter = [False, False, False, False], legend = ["Ni$_{n}$(PET)$_{2n}$", "Ni$_{6}$(PET)$_{12}$", "Ni$_{5}$(PET)$_{10}$", "Ni$_{4}$(PET)$_{8}$"])
# plt.yticks([])
# plot.plt.tight_layout()
# plot.uni_plot(test_deconv.residual_data.range((0, 1)), 2, x_window = (0, 1), scatter = True)



# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# plot.plt.show()
#im.show()






_x = np.array([i/1000 for i in range(4000)])
_y1 = ExGaussian(1, 0.5, 0.1, 0.000001).eval(_x)
_y2 = ExGaussian(1, 0.7, 0.1, 0.05).eval(_x)
_y3 = ExGaussian(1, 0.9, 0.1, 0.1).eval(_x)
_y4 = ExGaussian(1, 1.1, 0.1, 0.2).eval(_x)
_y5 = ExGaussian(1, 1.3, 0.1, 0.4).eval(_x)


exGauss1 = Data(_x, _y1, data_label="$\\tau$ = 0.00")
exGauss2 = Data(_x, _y2, data_label="$\\tau$ = 0.05")
exGauss3 = Data(_x, _y3, data_label="$\\tau$ = 0.10")
exGauss4 = Data(_x, _y4, data_label="$\\tau$ = 0.20")
exGauss5 = Data(_x, _y5, data_label="$\\tau$ = 0.40")

multi_plot(np.array([exGauss1, exGauss2, exGauss3, exGauss4, exGauss5]), fig_num=10, color_scheme=true_rainbow, linewidth=2)
plt.show()




# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/PET TLC Cropped.jpg")

# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# std = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))
#     std.append(np.std(data[i]))



# test = Data(np.array(x), np.array(y)).normalize()
# test.stds = np.array(std)
# # uni_plot(test, 30)

# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [55.1, 72.2, 91.7], x_window=(49, 106))
# #print(test_deconv)
# # #   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1)).flip()] , x_window = (0, 1), color_scheme = ["black"], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid"], scatter = [False], legend = ["Ni$_{n}$(PET)$_{2n}$"])
# plt.yticks([])
# plot.plt.tight_layout()
# plot.uni_plot(test_deconv.residual_data.flip(), 2, x_window = (0, 1), scatter = False)

# _x = np.array([i/100 for i in range(100)])
# _y1 = ExGaussian(1, 0.5, 2, 1).eval(_x)


# exGauss1 = Data(_x, _y1, data_label="$\tau$ = 1")
# multi_plot(np.array([exGauss1]), fig_num=10, color_scheme=rainbow)
# plt.show()
# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# plot.plt.show()


# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/CyPT Cropped4.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# std = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))
#     std.append(np.std(data[i]))



# test = Data(np.array(x), np.array(y)).normalize()
# test.stds = np.array(std)
# # uni_plot(test, 30)
# plt.show()
# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [29.9, 38, 49], x_window=(25, 61))
#print(test_deconv)
# #   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1)).flip()] + 
# [test_deconv.bands_data[i].range((0, 1)).flip() 
# for i in range(len(test_deconv.bands_data))] + [test_deconv.fit_data.range((0,1)).flip()], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "dashed", "dashed", "dashed", "dashed"], scatter = [False, False, False, False, False], legend = ["Ni$_{n}$(CyPT)$_{2n}$", "Ni$_{4}$(CyPT)$_{8}$", "Ni$_{5}$(CyPT)$_{10}$", "Ni$_{6}$(CyPT)$_{12}$", "Fit"])
# plt.yticks([])
# plot.plt.tight_layout()
# plot.uni_plot(test_deconv.residual_data.flip(), 2, x_window = (0, 1), scatter = False)



# for i in range(3):
#     plot.uni_plot(test_deconv.bands_data[i], 10 + i)
#     print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# plot.plt.show()


# im  = Image.open("/Users/shreyas/PycharmProjects/SpectroscoPy/SpectroscoPy/Research Data/CyHT Cropped.jpg")
# im = ImageOps.grayscale(im)

# width, height = im.width, im.height


# center = width//2

# im = im.crop((center-100, 0, center+100, height-100))

# height, width = im.size

# data = list(im.getdata())
# data = [data[i * width:(i + 1) * width] for i in range(height)]
# x = []
# y = []
# std = []
# for i in range(height):
#     x.append(i)
#     y.append(255 - np.mean(data[i]))
#     std.append(np.std(data[i]))



# test = Data(np.array(x), np.array(y)).normalize()
# test.stds = np.array(std)
# uni_plot(test, 30)
# # plt.show()
# test.data_label = "Ni$_{4}$(PET)$_{8}$"
# test_deconv = test.exgauss_deconvolute(units="nm", centers_given = [37.4, 65.3, 77.9], x_window=(28, 98))
#print(test_deconv)
#   91.48561        0.99354 4.2843  1.52379
# plot.multi_plot([test.range((0, 1)).flip()] + 
# [test_deconv.bands_data[i].range((0, 1)).flip() 
# for i in range(len(test_deconv.bands_data))] + [test_deconv.fit_data.range((0,1)).flip()], x_window = (0, 1), color_scheme = ["black"] + true_rainbow[1:4], xlabel = "Normalized Distance", ylabel = "Normalized Intensity", linestyle = ["solid", "dashed", "dashed", "dashed", "dashed"], scatter = [False, False, False, False, False], legend = ["Ni$_{n}$(CyHT)$_{2n}$", "Ni$_{4}$(CyHT)$_{8}$", "Ni$_{5}$(CyHT)$_{10}$", "Ni$_{6}$(CyHT)$_{12}$", "Fit"])
# plt.yticks([])
# plot.plt.tight_layout()
# plot.uni_plot(test_deconv.residual_data.flip(), 2, x_window = (0, 1), scatter = False)



for i in range(3):
    # plot.uni_plot(test_deconv.bands_data[i], 10 + i)
    print(test_deconv.bands[i].c, test_deconv.bands_data[i].area())

# plot.plt.show()