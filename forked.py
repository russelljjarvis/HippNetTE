
#import sys
import os
installs = ['neuron','mpi4py','xlrd','pyNN','seaborn','neo']
def install_deps(i):
  '''
  Hack in dependencies into to sys.path
  '''
  import os, sys
  if i not in sys.path:
    os.system('pip install '+str(i))
#_ = map(install_deps,installs)
#import os
_ = list(map(install_deps,installs))
#Compile NEUORN mod files.
#temp = os.getcwd()
#os.chdir('/opt/conda/lib/python3.5/site-packages/pyNN/neuron/nmodl')
#os.system('nrnivmodl')
#os.chdir(temp)


#wg = i
def child(i):
    qi_ascoli = None
    import qi_ascoli
    qi_ascoli.sim_runner(i)
    qi_ascoli = None
    print('done ', i)
    return 0

weight_gain_factors = {0.0025:None,0.0125:None,0.025:None,0.05:None,0.125:None,0.25:None,0.3:None,0.4:None,0.5:None,1.0:None,1.5:None,2.0:None,2.5:None,3.0:None}
#weight_gain_factors = {0:0.5,1:1.0,2:1.5,3:2.0,4:2.5:,5:3}

for i in weight_gain_factors.keys():
    pid = os.fork()
    child(i)
    print('done i',i)

exit()
