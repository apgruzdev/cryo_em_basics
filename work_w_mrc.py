import os

import numpy as np
import matplotlib.pyplot as plt

import mrcfile

path_2_mrcs = r'Box/_personal_data/em/test_data/10029/data/mrcs'
path_2_mrcs = os.path.join(os.path.expanduser('~'), path_2_mrcs)
first_file = os.listdir(path_2_mrcs)[0]
with mrcfile.open(os.path.join(path_2_mrcs, first_file)) as mrc:
    tmp_frame = mrc.data

plt.imshow(tmp_frame[0])
plt.colorbar()
plt.show()


