import numpy as np
import pandas as pd

class HestonModel:
    def __init__(self, kappa, theta, eta, rho, v0):
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.v0 = v0

    def characteristic_function(self, u, T, S0, r):
        pass

    

    