import numpy as np
from matplotlib import pyplot as plt
seg = 0.001
T = 10
class Plant:
    def __init__(self, init_state):
        self.M = np.array([0,0,2])
        self.s = init_state
        self.recorder = []

    def update(self, u):

        s = self.s
        R = np.matmul([[-2.3,-0.5,1.5],[0,-3,1],[0,0,-1]], s)
        ds = R + (self.M * u)
        self.s = self.s + ds*seg
        self.recorder.append(self.s)
        return self.s

    def plot(self):
        x = np.arange(0,len(self.recorder)) * seg
        plt.plot(x, self.recorder)
        plt.show()


if __name__ == '__main__':
    plant = Plant(np.array([1,1,2],dtype=np.float32))
    u = 2
    for i in range(int(T / seg)):
        plant.update(u)
        u=u*0.98

    plant.plot()