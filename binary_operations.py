from constants import DEFAULT_MODULO
import math
import random

def add_mod(x, y, p=DEFAULT_MODULO):
    return (x + y) % p

def product_mod(x, y, p=DEFAULT_MODULO):
    return (x * y) % p

def subtract_mod(x, y, p=DEFAULT_MODULO):
    return (x - y) % p

def divide_mod(x, y, p=DEFAULT_MODULO):
    return (x // y) % p

def add_square_mod(x, y, p=DEFAULT_MODULO):
    return (x**2 + y**2) % p

def factorial(x,y, p=DEFAULT_MODULO):
    return math.factorial(x)%p

def random_map(x,y,p=DEFAULT_MODULO):
    return random.randint(0,p)