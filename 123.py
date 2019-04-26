from functools import reduce
a = reduce(lambda x,y:x*y,[5,2,2]*3)
print(a)
