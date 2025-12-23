import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self):
        pass

    def price_options(self, S0, K, T, r, sigma, option_type):
        if option_type == 'CE':
            return self.price_call(S0, K, T, r, sigma)
        elif option_type == 'PE':
            return self.price_put(S0, K, T, r, sigma)
        else:
            raise ValueError("Option type must be 'CE' or 'PE'")

    def price_call(self, S0, K, T, r, sigma):
        if T <= 0:
            T = 1e-4  
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        #print(d1, d2, norm.cdf(d1), norm.cdf(d2))
        call_price = (S0 * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        #print("Call Price:", call_price)
        return call_price

    def price_put(self, S0, K, T, r, sigma):
        if T <= 0:
            T = 1e-4  
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        #print(d1, d2, norm.cdf(d1), norm.cdf(d2))
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))
        #print("Put Price:", put_price)
        return put_price