import os

import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import plotly
import plotly.graph_objects as go
import plotly.express as px

import mrcfile

from skimage import measure

# Plot as stamps

join_home = lambda path: os.path.join(os.path.expanduser('~'), path)
path_2_mrcs = join_home(r'Box/_personal_data/em/test_data/10029/data/mrcs')
path_2_picts = join_home(r'Box/_personal_data/em/test_data/10029/picts')

# +
rows = 20
cols = 10
dpi = 150

plt.ioff()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                    wspace=0.02, hspace=0.02)

for ind, file_name in tqdm.tqdm(enumerate(os.listdir(path_2_mrcs))):
    with mrcfile.open(os.path.join(path_2_mrcs, file_name)) as mrc:
        data_frame = mrc.data
        fig = plt.figure(figsize=(cols*2, rows*2))
        for frame_ind, frame in enumerate(data_frame):
            fig.add_subplot(rows, cols, frame_ind+1)
            plt.imshow(frame)
            plt.axis('off')
        plt.savefig(os.path.join(path_2_picts, f'{ind}.png'), dpi=dpi)
        plt.close(fig)


# -
# Plot 3d

def matrix_3d_2_xyz(arr_3d: np.ndarray) -> pd.DataFrame():
    vals = arr_3d.reshape(-1)
    z, y, x = np.indices(arr_3d.shape)
    df = pd.DataFrame(np.array([x.reshape(-1), y.reshape(-1), z.reshape(-1), vals]).T, columns=['x', 'y', 'z', 'vals'])
    return df


# path_2_src = r'/Users/fawcettpercival/Documents/GitHub/cryoem-cvpr2015/Data/1AON/imgdata.mrc'
# path_2_3d_model = r'/Users/fawcettpercival/Documents/GitHub/cryoem-cvpr2015/exp/1AON_sagd_is/dmodel.mrc'
path_2_3d_model = r'/Users/fawcettpercival/Documents/GitHub/cryoem-cvpr2015/exp/1AON_sagd_is/model.mrc'

df = None
data_frames = None
with mrcfile.open(path_2_3d_model) as mrc:
    data_frames = mrc.data
    df = matrix_3d_2_xyz(data_frames)
    df = df[df['vals'] > 1e-3]

# +
fig = go.Figure(data=[go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['z'],
    mode='markers',
    marker=dict(symbol='square', size=2, colorscale='rdbu', opacity=0.3, color=df['vals'])
)])

plotly.offline.plot(fig, filename=r'/Users/fawcettpercival/Documents/GitHub/cryoem-cvpr2015/exp/1AON_sagd_is/3d_model.html', 
                    show_link=True, auto_open=True)
# -

one_frame = data_frames[50]

fig = plt.figure()
plt.imshow(one_frame)
plt.axis('off')
fig.show()

one_frame_bin = ((np.abs(one_frame) > 0.) * 1)

fig = plt.figure()
plt.imshow(one_frame_bin)
plt.axis('off')
fig.show()

# contours = measure.find_contours(one_frame, level=np.median(one_frame[one_frame > 0.]))
contours = measure.find_contours(one_frame_bin, level=.5)

# +
fig, ax = plt.subplots()
ax.imshow(one_frame, cmap=plt.cm.gray)

# for n, contour in enumerate(contours):
#     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
contours_concat = np.concatenate(contours)
ax.scatter(contours_concat[:, 1], contours_concat[:, 0], s=2)
    
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# +
contour_frames = []
for ind, frame in enumerate(data_frames):
    frame_bin = ((np.abs(frame) > 0.) * 1)
    frame_contours = measure.find_contours(frame_bin, level=.5)
    frame_contours_concat = np.concatenate(frame_contours)
    contour_frame = np.concatenate([frame_contours_concat, ind * np.ones((frame_contours_concat.shape[0], 1))], axis=1)
    contour_frames.append(contour_frame)

contiur_df = pd.DataFrame(np.concatenate(contour_frames), columns=['x', 'y', 'z'])

# +
fig = go.Figure(data=[go.Scatter3d(
    x=contiur_df['x'],
    y=contiur_df['y'],
    z=contiur_df['z'],
    mode='markers',
    marker=dict(size=1, colorscale='rdbu', opacity=0.5)
)])

plotly.offline.plot(fig, filename=r'/Users/fawcettpercival/Documents/GitHub/cryoem-cvpr2015/exp/1AON_sagd_is/3d_model_contours.html', 
                    show_link=True, auto_open=True)
# -
# Fourier research


from scipy.fftpack import fft2, ifft2, fftshift

spectrum_one_frame = fft2(one_frame)

spectrum_one_frame[0, 0]

fig = plt.figure()
plt.imshow(np.abs(spectrum_one_frame), norm=LogNorm(), cmap=plt.cm.gray)
# plt.imshow(np.abs(ifft2(spectrum_one_frame)))
plt.axis('off')
fig.show()

fig = plt.figure()
plt.imshow(np.abs(fftshift(spectrum_one_frame)), norm=LogNorm(), cmap=plt.cm.gray)
# plt.imshow(np.abs(ifft2(fftshift(spectrum_one_frame))))
plt.axis('off')
fig.show()

import os
os.makedirs('rrr/ttt')

os.path.exists(os.path.dirname('tt.yy'))

os.path.dirname('tt.yt')


