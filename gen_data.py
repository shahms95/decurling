import numpy as np
from scipy.misc import imread
import subprocess
import random

cage_no = 15
image_no = 2940

cages = np.zeros((2*cage_no+2, 16, 3))

for ind in range(1, cage_no+1):
    cages[ind-1] = np.genfromtxt('GenData/cage'+str(ind)+'.txt')
    cages[cage_no+ind] = np.genfromtxt('GenData/cage'+str(ind)+'r.txt')

# images = np.zeros((image_no, ))

# for ind in range(1, image_no+1):
#     images[ind] = imread('papers-'+str(ind)+'.png')

to_gen = 20000


def getrandomcage():
    fir = random.randint(0, 2*cage_no+1)
    sec = random.randint(0, 2*cage_no+1)
    while fir == sec:
        sec = random.randint(0, 2*cage_no+1)

    ifir = cages[fir]
    isec = cages[sec]

    p = random.uniform(0.0, 1.0)
    ifin = ifir*p + isec*(1-p)
    return ifin

for ind in range(1, to_gen+1):
    randcage = getrandomcage()
    np.savetxt('Data/labels/tmp'+str(ind)+'.txt', randcage)
    randpaper = random.randint(0, image_no+1)
    # subprocess.call(['Build/Output/bin/RenderPage', '-c', /*path/to/cage/file*/, /*/path/to/input/image*/, /*/path/to/output/image*/, '648', '1152']) //see next line for concrete example
    subprocess.call(['Build/Output/bin/RenderPage', '-c', 'Data/labels/tmp'+str(ind)+'.txt', '../datasets/all/'+str(randpaper)+'.png', 'Data4/images/out'+str(ind)+'.png', '648', '1152'])
