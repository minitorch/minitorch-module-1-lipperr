"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -1. * x

def lt(x: float, y: float) -> float:
    return 1. if x < y else 0.

def eq(x: float, y: float) -> float:
    return 1. if x == y else 0.

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return 1. if abs(x-y) < 1e-2 else 0.

def sigmoid(x: float) -> float:
    return 1. / (1.0 + math.exp(-x)) if x >=0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    return max(0., x)

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def inv(x: float) -> float:
    return 1. / x

def log_back(x: float, y: float) -> float:
    return inv(x) * y

def inv_back(x: float, y: float) -> float:
    return -inv(x * x) * y

def relu_back(x: float, y: float) -> float:
    return int(x >= 0) * y


def map(func: Callable, arr: Iterable) -> Iterable:
    return [func(x) for x in arr]

def zipWith(func: Callable, arr1: Iterable, arr2: Iterable) -> Iterable:
    return [func(x, y) for x, y in zip(arr1, arr2)]

def reduce(func: Callable, arr: Iterable) -> float:
    x = arr[0] if len(arr) > 0 else 0
    for y in arr[1:]:
        x = func(x, y)
    return x

def negList(arr: Iterable) -> Iterable:
    return map(neg, arr)

def addLists(arr1: Iterable, arr2: Iterable) -> Iterable:
    return zipWith(add, arr1, arr2)

def sum(arr: Iterable) -> float:
    return reduce(add, arr)

def prod(arr: Iterable) -> float:
    return reduce(mul, arr)