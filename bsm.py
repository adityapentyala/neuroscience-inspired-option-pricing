import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, S0, K, T, r, C):
        self.S0 = S0  # Current stock price
        self.K = K    # Strike price        
        self.T = T    # Time to maturity (in years)
        self.r = r    # Risk-free interest rate
        self.C = C    # Option type: 'call' or 'put'

    def price_call(self, S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = (S0 * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        return call_price

    def price_put(self, S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))
        return put_price