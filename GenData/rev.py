import sys
import numpy as np
f1 = open("cage"+sys.argv[1]+".txt",'r')
f2 = "cage"+sys.argv[1]+"r.txt"
a = np.loadtxt(f1)
for i in xrange(4):
	a[4*i][2],a[4*i+3][2] = a[4*i+3][2],a[4*i][2]
	a[4*i+1][2],a[4*i+2][2] = a[4*i+2][2],a[4*i+1][2]
np.savetxt(f2, a)