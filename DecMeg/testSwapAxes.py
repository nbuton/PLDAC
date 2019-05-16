#(trial x channel x time) of size 580 x 306 x 375.
"""
[
trial 1 : [
channel 1 : [t1 : 1 , t2: 2]
channel 2 : [t1 : 3, t2: 4]
channel 3 : [t1 : 5 , t2: 6 ]
]
trial 2 : [
channel 1 : [t1 : 7, t2:8]
channel 2 : [t1 : 9, t2: 10]
channel 3 : [t1 :11, t2:12]
]
]
2x3x2
debut :
[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]
resultat attendu :
[ [ [1,3,5],[2,4,6] ], [ [7,9,11],[8,10,12] ] ]

"""
import numpy as np
tab1 = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
tab2 = np.swapaxes(tab1,2,1)
print(tab1)
print(tab2)
