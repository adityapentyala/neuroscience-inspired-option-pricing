import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq, newton

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
    
    def implied_volatility(self, market_price, S0, K, T, r, option_type, lower = 1e-4, upper = 5, tol=1e-6):
        def objective_function(sigma):
            #print("Trying sigma:", sigma)
            model_price = self.price_options(S0, K, T, r, sigma, option_type)
            return model_price - market_price
        def d_objective_function(sigma):
            #print("Calculating Vega for sigma:", sigma)
            return self.calculate_vega(S0, K, T, r, sigma, option_type)
        try:
            result = newton(objective_function, x0=0.15, tol=tol, maxiter=100)
        except RuntimeError as e:
            print("Newton's method failed:", e)
            try:
                result = brentq(objective_function, a=lower, b=upper, xtol=tol, maxiter=100)
            except ValueError as ve:
                print("Brent's method failed:", ve)
                result = np.nan
        return result


    def calculate_delta(self, S0, K, T, r, sigma, option_type):
        if T <= 0:
            T = 1e-4 
        if option_type == 'CE':
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1)
        elif option_type == 'PE':
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            delta = -norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'CE' or 'PE'")
        return delta
    
    def calculate_gamma(self, S0, K, T, r, sigma, option_type):
        if T <= 0:
            T = 1e-4 
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        return gamma
    
    def calculate_vega(self, S0, K, T, r, sigma, option_type):
        if T <= 0:
            T = 1e-4 
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S0 * norm.pdf(d1) * np.sqrt(T)
        return vega
    
    def calculate_theta(self, S0, K, T, r, sigma, option_type):
        if T <= 0:
            T = 1e-4 
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'CE':
            theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'PE':
            theta = -S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Option type must be 'CE' or 'PE'")
        return theta
    
    def calculate_rho(self, S0, K, T, r, sigma, option_type):
        if T <= 0:
            T = 1e-4 
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'CE':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'PE':
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Option type must be 'CE' or 'PE'")
        return rho