# библиотеки
from math import *
import numpy as np
from scipy.optimize import bisect
from scipy.integrate import quad


class Star:
    def __init__(self, D, e_0, rho_t,kappa_0_max, n, beta, r):
        self.D = D
        self.e_0 = e_0
        self.n = n
        self.beta = beta
        self.r = r
        self.rho_t = rho_t
        self.kappa_0_max = kappa_0_max  # Сохраняем максимальное значение
        self.kappa_0 = kappa_0_max      # Начальное значение
        self.alfa = pi / n
        self.R = self.D/2 - self.e_0 - self.r     
        self.gamma_1 = pi/2 + self.alfa - self.beta
        
        self.e_0_dim = 2*self.e_0/self.D
        self.r_dim = self.r/self.R
        self.R_dim = 2*self.R/self.D
        
        # Расчет производных параметров
        self.e_1 = self.R*(sin(self.alfa)/cos(self.beta)) - self.r
        self.e_1_dim = 2*self.e_1/self.D
        
        # ОСНОВНАЯ ОШИБКА: исправлен расчет e_max
        self.e_max = sqrt(self.R**2 + (self.D/2)**2 - self.R*self.D*cos(self.alfa)) - self.r
        self.e_max_dim = 2*self.e_max/self.D  # Исправлено
        
        self.F_ks = pi*self.D**2/4
        
        # Расчет начальных параметров
        self.eps_f = self.calculate_eps_f()
        # Сначала вычисляем l, так как оно нужно для S_g
        self.l = (self.kappa_0 * self.F_ks * (1 - self.eps_f)) / ( self.P(0))
        self.omega_t =  self.l * (self.rho_t * self.eps_f * self.F_ks)
        self.W_zar = self.omega_t/self.rho_t        
        
        self.xi_degr = self.calculate_xi_degr()
        self.s_sr0 = self.calculate_s_sr0()
        self.s_sr = self.calculate_s_sr()
    
    def b(self, e):
        return self.R*(sin(self.alfa)/sin(self.beta)) - (self.r + e)*(1/tan(self.beta))
    
    def gamma(self, e):
        # Исправлено: добавлена проверка на область определения asin
        arg = self.R*sin(self.alfa)/(self.r + e)
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return self.alfa + asin(arg)
    
    def delta(self, e):
        # Исправлено: добавлена проверка на область определения acos
        arg = (self.R**2 + (self.r + e)**2 - (self.D/2)**2) / (2*self.R*(self.r + e))
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return pi - acos(arg)
    
    def phi(self, e):
        # Исправлено: добавлена проверка на область определения acos
        arg = (self.R**2 + (self.D/2)**2 - (self.r + e)**2) / (self.R*self.D)
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return acos(arg)
    
    def P(self, e):
        if e < 0:
            return 0
        elif e <= self.e_1:
            return 2*self.n*(self.gamma_1*(self.r + e) + self.b(e))
        elif e <= self.e_0:
            return 2*self.n*self.gamma(e)*(self.r + e)
        elif e <= self.e_max:
            return 2*self.n*(self.gamma(e) - self.delta(e))*(self.r + e)
        else:
            return 0
    
    def S_g(self, e):
        return self.P(e) * self.l
    
    def F_kan(self, e):
        if e < 0:
            return 0
        elif e <= self.e_1:
            return self.n*(self.gamma_1 * (self.r + e)**2 + 
                          (self.R * sin(self.alfa)/sin(self.beta) - 
                           (self.r + e)*(1/tan(self.beta)))*(self.r + e + 
                           self.R*cos(self.gamma_1)) + 
                           self.R*(self.r + e)*sin(self.gamma_1))
        elif e <= self.e_0:
            return self.n*(self.gamma(e)*(self.r + e)**2 + 
                          self.R*(self.r + e)*sin(self.gamma(e)))
        elif e <= self.e_max:
            return self.n*((self.gamma(e) - self.delta(e))*(self.r + e)**2 + 
                          self.R*(self.r + e)*sin(self.gamma(e)) + 
                          self.phi(e)*self.D**2/4 - 
                          self.R*self.D/2*sin(self.phi(e)))
        else:
            return 0
    
    def calculate_eps_f(self):
        term = (self.gamma_1*self.r**2 + 
                (self.R*sin(self.alfa)/sin(self.beta) - 
                 self.r*(1/tan(self.beta)))*(self.r + self.R*cos(self.gamma_1)) + 
                self.R*self.r*sin(self.gamma_1))
        return 1 - (4*self.n/(pi*self.D**2)) * term 
    
    def kappa(self,e):
        F_kan_val = self.F_kan(e)
        if F_kan_val == 0:
            return float('inf')
        return (self.S_g(e)/self.F_kan(e))
    
    def calculate_kappa_0(self):
        F_kan_0 = self.F_kan(0)
        if F_kan_0 == 0:
            return float('inf')
        return self.S_g(0) / F_kan_0  # Исправлено
    
    def calculate_xi_degr(self):
        F_kan2 = self.F_kan(self.e_0)
        return 1/self.eps_f * (1 - 4 * F_kan2/(pi*self.D**2))
    
    def calculate_s_sr0(self):
        def integrand(e):
            return self.S_g(e)
        
        integral, error = quad(integrand, 0, self.e_0)
        return (1 / self.e_0) * integral 
    
    
    def calculate_s_sr(self):
        def integrand(e):
            return self.S_g(e)
        
        integral, error = quad(integrand, 0, self.e_max)
        return (1 / self.e_max) * integral

    def get_params(self):
        """Возвращает словарь со всеми вычисленными параметрами"""
        return {
            'eps_f': self.eps_f,
            'omega_t': self.omega_t,
            'l': self.l,
            'xi_degr': self.xi_degr,
            'kappa_0': self.kappa_0,
            'e_0_dim': self.e_0_dim,
            'R_dim': self.R_dim,
            'r_dim': self.r_dim,
            'e_max_dim': self.e_max_dim,
            'F_ks': self.F_ks,
            's_sr0': self.s_sr0,
            's_sr': self.s_sr
        }


class Star_2:
    def __init__(self, D, e_0, omega, rho_t, n, bet, r):
        self.D = D
        self.e_0 = e_0
        self.n = n
        self.bet = bet
        self.r = r
        self.omega = omega
        self.rho_t = rho_t
        self.alfa = pi / n
        self.R = self.D/2 - self.e_0 - self.r     
        self.gamma_1 = pi/2 + self.alfa - self.bet
        self.W_zar = self.omega/self.rho_t
        self.e_0_dim = 2*self.e_0/self.D
        self.r_dim = self.r/self.R
        self.R_dim = 2*self.R/self.D
        
        # Расчет производных параметров
        self.e_1 = self.R*(sin(self.alfa)/cos(self.bet)) - self.r
        self.e_1_dim = 2*self.e_1/self.D
        

        self.e_max = sqrt(self.R**2 + (self.D/2)**2 - self.R*self.D*cos(self.alfa)) - self.r
        self.e_max_dim = 2*self.e_max/self.D  # Исправлено
        
        self.F_ks = pi*self.D**2/4
        
        # Расчет начальных параметров
        self.eps_f = 1 - self.F_kan(0)/self.F_ks
        # Сначала вычисляем l, так как оно нужно для S_g
        self.l = self.omega / (self.rho_t * self.eps_f * self.F_ks)
        self.kappa_0 = self.calculate_kappa_0()
        self.xi_degr = self.calculate_xi_degr()
        self.s_sr0 = self.calculate_s_sr0()
        self.s_sr = self.calculate_s_sr()
        

    
    def b(self, e):
        return self.R*(sin(self.alfa)/sin(self.bet)) - (self.r + e)*(1/tan(self.bet))
    
    def gamma(self, e):
        # Исправлено: добавлена проверка на область определения asin
        arg = self.R*sin(self.alfa)/(self.r + e)
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return self.alfa + asin(arg)
    
    def delta(self, e):
        # Исправлено: добавлена проверка на область определения acos
        arg = (self.R**2 + (self.r + e)**2 - (self.D/2)**2) / (2*self.R*(self.r + e))
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return pi - acos(arg)
    
    def phi(self, e):
        # Исправлено: добавлена проверка на область определения acos
        arg = (self.R**2 + (self.D/2)**2 - (self.r + e)**2) / (self.R*self.D)
        if arg > 1: arg = 1
        elif arg < -1: arg = -1
        return acos(arg)
    
    def P(self, e):
        if e < 0:
            return 0
        elif e <= self.e_1:
            return 2*self.n*(self.gamma_1*(self.r + e) + self.b(e))
        elif e <= self.e_0:
            return 2*self.n*self.gamma(e)*(self.r + e)
        elif e <= self.e_max:
            return 2*self.n*(self.gamma(e) - self.delta(e))*(self.r + e)
        else:
            return 0
    
    def S_g(self, e):
        return self.P(e) * self.l
    
    # В классе Star исправить:
    def F_kan(self, e):
        if e < 0:
            return 0
        elif e <= self.e_1:
            return self.n*(self.gamma_1 * (self.r + e)**2 + 
                          (self.R * sin(self.alfa)/sin(self.bet) - 
                           (self.r + e)*(1/tan(self.bet)))*(self.r + e + 
                           self.R*cos(self.gamma_1)) + 
                           self.R*(self.r + e)*sin(self.gamma_1))
        elif e <= self.e_0:
            return self.n*(self.gamma(e)*(self.r + e)**2 + 
                          self.R*(self.r + e)*sin(self.gamma(e)))
        elif e <= self.e_max:
            # ИСПРАВЛЕНИЕ: использовать self.D/2 вместо self.D
            return self.n*((self.gamma(e) - self.delta(e))*(self.r + e)**2 + 
                          self.R*(self.r + e)*sin(self.gamma(e)) + 
                          self.phi(e)*(self.D/2)**2 -  # ИСПРАВЛЕНО
                          self.R*(self.D/2)*sin(self.phi(e)))  # ИСПРАВЛЕНО
        else:
            return 0
    
    def calculate_eps_f(self):
        term = (self.gamma_1*self.r**2 + 
                (self.R*sin(self.alfa)/sin(self.bet) - 
                 self.r*(1/tan(self.bet)))*(self.r + self.R*cos(self.gamma_1)) + 
                self.R*self.r*sin(self.gamma_1))
        return 1 - (4*self.n/(pi*self.D**2)) * term  # Исправлено: добавлен множитель n
    
    def kappa(self,e):
        F_kan_val = self.F_kan(e)
        if F_kan_val == 0:
            return float('inf')
        return (self.S_g(e)/self.F_kan(e))
    
    def calculate_kappa_0(self):
        F_kan_0 = self.F_kan(0)
        if F_kan_0 == 0:
            return float('inf')
        return self.S_g(0) / F_kan_0  # Исправлено
    
    def calculate_xi_degr(self):
        F_kan2 = self.F_kan(self.e_0)
        return 1/self.eps_f * (1 - 4 * F_kan2/(pi*self.D**2))
    
    def calculate_s_sr0(self):
        def integrand(e):
            return self.S_g(e)
        
        integral, error = quad(integrand, 0, self.e_0)
        return (1 / self.e_0) * integral  
    
    
    def calculate_s_sr(self):
        def integrand(e):
            return self.S_g(e)
        
        integral, error = quad(integrand, 0, self.e_max)
        return (1 / self.e_max) * integral
    
    def get_params_2(self):
        """Возвращает словарь со всеми вычисленными параметрами"""
        return {
            'eps_f': self.eps_f,
            'omega_t': self.omega,
            'l': self.l,
            'xi_degr': self.xi_degr,
            'kappa_0': self.kappa_0,
            'e_0_dim': self.e_0_dim,
            'R_dim': self.R_dim,
            'r_dim': self.r_dim,
            'e_max_dim': self.e_max_dim,
            'F_ks': self.F_ks,
            's_sr0': self.s_sr0,
            's_sr': self.s_sr,
            'e_1':self.e_1,
            'e_max':self.e_max,
            'S_g(e)':self.S_g,
            'kappa(e)':self.kappa,
        }
    