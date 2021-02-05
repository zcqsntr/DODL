import math

def find_unique_lgs(n):

    n_IS = 2**n

    total = math.factorial(n_IS + 1)
    for i in range(n_IS+1):
        total -= (math.factorial(i)*math.factorial(n_IS-i) - 1)*math.factorial(n_IS)/(math.factorial(n_IS-i)*math.factorial(i))

    return total


def lgs_position(n):

    n_IS = 2**n
    for i in range(n_IS+1):
        print(math.factorial(n_IS)/(math.factorial(n_IS-i)*math.factorial(i)))

print(find_unique_lgs(2))
print(lgs_position(2))
print(find_unique_lgs(3))
print(find_unique_lgs(4))