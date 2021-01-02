import numpy as np
import scipy.stats as sts
import pandas as pd
import scipy, math
import matplotlib.pyplot as plt
import itertools

data = pd.read_csv('datafile.csv')
# The data file was created by varying 'a' in mle.py until the value of lambda was 2.714.
# A virtual machine from Amazon Web Services was used to find the values of 'a'. 
# The values of 'a' were recorded for different values of energy.
# Note that the values of 'a' are recorded in the y-axis (labelled as sigma_v).

y = np.array(data['sigmav'], dtype=float)
x = np.array(data['energy'], dtype=int)

plt.scatter(x,y)
plt.yscale('log')
plt.ylim([10**-27,10**-22])
plt.xlim([1,100])
plt.xscale('log')
plt.xlabel('Energy(TeV)')
plt.ylabel('sigma_v')
plt.savefig('FinalGraph.pdf')
