import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift
from skimage.transform import radon, iradon
from scipy.spatial import distance_matrix
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

h = 100  # picture size
center = np.array([h/2., h/2.])


def simple_plot(canvas):
    fig = plt.figure()
    plt.imshow(canvas, cmap=plt.cm.gray)
    fig.show()


# cercle

circle = np.zeros((h, h))

r = h / 4.  # radius
for alpha in np.arange(0, 2.*np.pi, np.pi/90.):
    coord = (center + r * np.array([np.cos(alpha), np.sin(alpha)])).astype(int)
    circle[coord[0]][coord[1]] = 1.

simple_plot(circle)

# ellipse

ellipse = np.zeros((h, h))

r = np.array([h / 8., h / 4.])  # width and height 
for alpha in np.arange(0, 2.*np.pi, np.pi/90.):
    coord = (center + r * np.array([np.cos(alpha), np.sin(alpha)])).astype(int)
    ellipse[coord[0]][coord[1]] = 1.

simple_plot(ellipse)

ellipse_rotated = (rotate(ellipse, angle=30, reshape=False) > 0.3).astype(int)

simple_plot(ellipse_rotated)

# Fourier

circle_fft = fft2(circle)
ellipse_fft = fft2(ellipse)

simple_plot(np.abs(fftshift(circle_fft)))

simple_plot(np.abs(fftshift(ellipse_fft)))

# Sinogram / Radon transform

thetas = np.linspace(0., 180., num=180, endpoint=False)
# As a rule of thumb, the number of projections should be about the same as the number of pixels there are across the object (num=h*h)

circle_sinogram = radon(circle, theta=thetas)

simple_plot(circle_sinogram)

# ellipse_sinogram = radon(ellipse, theta=thetas)
ellipse_sinogram = radon(ellipse_rotated, theta=thetas)

simple_plot(ellipse_sinogram)

sinogram_dist = distance_matrix(circle_sinogram.T, ellipse_sinogram.T)

simple_plot(sinogram_dist)

applicable_angles = np.argwhere(np.abs(sinogram_dist - sinogram_dist.max()) < 1e-3)
applicable_angle = applicable_angles[0]

# Plots with found lines

ellipse_w_line = ellipse_rotated.copy()

for x in range(h):
    rel_x = x - h / 2.
    rel_y = np.tan(np.radians(applicable_angle[1])) * rel_x
    y = (rel_y + h / 2.).astype(int)
    if (y < 0) or (y > h-1):
        continue
    ellipse_w_line[x][y] = 1.

simple_plot(ellipse_w_line)

b = [1, 2, 3]
def a():
    return next(t for t in b)


a()

import pandas as pd
pd.isnull(np.nan), pd.isnull(None)

pd.isnull(None)

None is None

np.nan

np.NaN


