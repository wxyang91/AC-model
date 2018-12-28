import numpy as np
from env import M
from env import Plant
from matplotlib import pyplot as plt

class observer:
    def __init__(self, dim):
        self.B = np.mat(M)
        self.est_x = np.mat(np.array([0., 0., 0., 0.])).T
        self.est_w = np.mat(np.zeros([dim])).T
        self.est_B = np.mat(np.zeros([4,4]))

    def __call__(self, *args, **kwargs):
        rx = np.mat(kwargs['rx'])
        x = np.mat(kwargs['x'])
        u = np.mat(kwargs['u'])
        L1 = np.mat(np.diag([5., 1., 1., 5.]))
        L2 = np.mat(np.diag([5., 1., 1., 5.]))
        ex = x - self.est_x
        dx = self.est_w + rx + self.est_B * u.T + L1 * ex
        dw = np.mat(L2) * np.mat(L1).I * ex
        nu = u * u.T
        L3 = float(nu)
        dB = 0.004 * np.mat(L1).I * ex * u
        eB = self.B - self.est_B
        dd = np.mat(np.ones([4])) * (eB.T * eB) * np.mat(np.ones([4])).T

        self.est_x += dx * 0.001
        self.est_w += dw * 0.001
        self.est_B += dB * 0.01

if __name__ == '__main__':
    obs = observer(4)
    plant = Plant(np.array([1, 2, 2, 3], dtype=np.float32))
    u = np.ones([4])
    xs = []
    x=None
    for i in range(5000):
        if i % 10 == 0:
            x = plant.update(u)
            du = -0.2 * u
            u = u + du
            x = np.mat(x).T
        r = np.mat(np.diag([-12.1,-2.2,-2.2,-2.2])) * x
        obs(rx=r, x = x, u=u)
        est_x = obs.est_w.copy()
        xs.append(est_x)
    print obs.est_B

    x = np.arange(0, len(xs))
    xs = np.reshape(np.array(xs),(5000, 4))
    #plt.plot(x,xs[:,3])
    plt.plot(x, xs[:, 0])
    plt.show()
