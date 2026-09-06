from environment.grenight_environment import GrenightEnvironment

import numpy as np
np.set_printoptions(threshold=np.inf)

env = GrenightEnvironment(True, True, False)
env.reset()
print([[p.uid, p.position] for p in env.pieces])
print(env.get_state())
env.step(env.sample())
env.step(env.sample())
env.step(env.sample())
print([[p.uid, p.position] for p in env.pieces])
print(env.get_state())

