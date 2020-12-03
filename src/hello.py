print('Hello World!')
print('2nd Test')



d = dict()
d['a'] = dict()
d['a']['b'] = 5
d['a']['c'] = 6
d['x'] = dict()
d['x']['y'] = 10
print(d)

print(d['a'])


import random
random.seed(30)

r = random.randrange(0,5)
print(r)


import numpy as np
np.random.seed
for i in range(20):
    newArray = list(set(np.random.random_integers(0, 10, size=(6))))[:3]
    print(newArray)