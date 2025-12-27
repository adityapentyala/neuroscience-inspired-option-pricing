import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.express as px

class MonteCarloModel:
    def __init__(self, volatility_approx='gbm'):
        self.volatility_approx = volatility_approx

    def price_option(self, S0, K, sigma, r, T, steps=50, n_simulations=1000, option_type='CE'):
        #dlnSt = nu*dt + sigma*dW
        dt = T / steps
        nu = r - 0.5 * sigma**2
        lnS0 = np.log(S0)

        if self.volatility_approx == 'gbm':
            Z = np.random.normal(0, 1, (steps, n_simulations))

            dlnSt = nu * dt + sigma * np.sqrt(dt) * Z
            lnSt = lnS0 + np.cumsum(dlnSt, axis=0)

            ST = np.exp(lnSt[-1]) 

            if option_type == 'CE':
                payoffs = np.maximum(ST - K, 0)
            elif option_type == 'PE':
                payoffs = np.maximum(K - ST, 0)
            else:
                raise ValueError("option_type must be 'CE' or 'PE'")

            discounted_payoffs = np.exp(-r * T) * payoffs

            option_price = np.mean(discounted_payoffs)

            sample_std = np.std(discounted_payoffs, ddof=1)
            stderr = sample_std / np.sqrt(n_simulations)
            
            return option_price, stderr

        else:
            raise NotImplementedError("Only 'gbm' volatility approximation is implemented.")