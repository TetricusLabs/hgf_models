from re import M
import numpy as np
from softmax_conf import *
from types import SimpleNamespace

inf_states = np.random.rand(160, 3, 5, 5)
# inf_states = np.full((160, 3, 5, 5), -1, dtype=np.int8)


r = SimpleNamespace()

r.y = np.zeros((160, 1))
r.u = np.zeros((160, 1))
r.ign = []
r.irr = []
r.c_prc = {}
x = SimpleNamespace()
x.predorpost = 1
x.model = 'softmax_mu3'
x.priormus = []
x.priorsas = []
r.c_obs = x


tapas_softmax_mu3(r, inf_states)
