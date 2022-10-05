from re import M
import numpy as np
from softmax_conf import *
from types import SimpleNamespace

inf_states = np.random.rand(160, 3, 5, 4)
# inf_states = np.full((160, 3, 5, 5), -1, dtype=np.int8)


r = {}

r['y'] = np.zeros((160, 1))
r['u'] = np.zeros((160, 1))
r['ign'] = np.array([])
r['irr'] = np.array([])
r['c_prc'] = {}
# x = SimpleNamespace()
x = {}
x['predorpost'] = 1
x['model'] = 'softmax_mu3'
x['priormus'] = np.array([])
x['priorsas'] = np.array([])
r['c_obs'] = x


tapas_softmax_mu3(r, inf_states)
