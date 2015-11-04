import theano
from theano import tensor as T

a = T.scalar()
b = T.scalar()

y = a * b

multiply = theano.function(inputs=[a, b], outputs=y)


def multiply_func(x, y):
    return multiply(x, y)

# print(multiply(1, 2))  # 3
# print(multiply(3, 3))  # 9
