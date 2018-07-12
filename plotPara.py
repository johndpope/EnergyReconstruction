import matplotlib.pyplot as plt
import numpy as np
from random import random
from scipy.optimize import leastsq, curve_fit, fmin
from scipy.spatial import ConvexHull
import warnings
import sys, ast
np.set_printoptions( precision = 5 )

def parabola( x, a, b, c ):
    return a + b * x + c * x**2

### find the shortest distance between a given point and a parabola
### actually give x such as ( x, f(x) ) is closest to ( x0, y0 )
def orth_dist( x0, y0, a, b, c ):
    f = lambda x: np.sqrt( ( x - x0 )**2 + ( parabola( x, a, b, c ) - y0 )**2 )
    sol = fmin( f, x0, disp=False )
    return sol[ 0 ]

### cost function: standard distance if point is above parabola
### or extra cost if point is below
def myDist( r, dy , off=4e0 ):
    out =  abs( r )
    if dy <= 0:
        out = max( [out, off + off * abs( r)  ] )
    return out

def residuals( p, data, off=0 ): # data is of type [ [x0,y0], [x1,y1],... ] 
    xList, yList = zip( *data )
    ### a standard fit uses the distance in y direction between curve and 
    ### data point. Here it is better to use the shortest distance.
    ### To find the shortest distance some extra numerics is required.
    xopt = [ orth_dist( x0, y0, *p ) for x0, y0 in zip( xList, yList ) ]
    yopt = [ parabola( x, *p ) for x in xopt ]
    yFitList = [ parabola( x, *p ) for x in xList ]
    rList = [ np.sqrt( ( x - x0 )**2 + (y - y0 )**2 ) for x, x0, y , y0 in zip( xList, xopt, yList, yopt ) ]
    signList = [ y - yTh  for y, yTh in zip( yList, yFitList ) ]
    out = [ myDist( r, dy , off=off ) for r, dy in zip( rList, signList ) ]
    return out

### removing the upper points of the convex hull
### as it is convex, it can be at most a straight line 
### from the most left to the most right point
### if a point of the hull is on or above this line, dump it
def remove_upper( data ):
    sortedByX = sorted( data)
    xList, yList = zip( *sortedByX )
    m = ( yList[-1] - yList[0] ) / ( xList[-1] - xList[0] )
    out = [ sortedByX[0] ]
    for p in range( 1, len(data ) ):
        x = xList[ p ]
        y = yList[ p ]
        if y < m * ( x - xList[0]) + yList[0]:
            out += [ sortedByX[ p ] ]
    out += [ sortedByX[ -1 ] ]
    return out

### my choice of 'true' parameters
myParameters = ( .45, -1.3, 2.33)

### just some random data for testing
#scatterData = list()
#for i in range(180):
#    x = 3*( 2 * random() - 1) + 1.1
#    y = 8 * random()
#    if y > parabola( x, *myParameters ):
#        scatterData += [ [ x, y ] ]
scatterData = sys.argv[1]
scatterData = ast.literal_eval(scatterData)

### to fit the parabula we do not need inner points, 
### so let's take the convex hull
hull = ConvexHull(scatterData )
scatterDataHull = list()
for index in hull.vertices:
    scatterDataHull += [ scatterData[ index ] ]

### the upper points of the convex hull are also useless 
#### if not disturbing, so get rid of them as well 
scatterDataHullR = remove_upper( scatterDataHull )

### exact data for plotting
xxList = np.linspace( -4, 4, 300 )
yyList = [ parabola( x, *myParameters ) for x in xxList ]

scatterX, scatterY =zip(*scatterDataHullR)
try:
    start, pcov = curve_fit( parabola,  *zip(*scatterDataHullR) )
except:
    start, pcov = curve_fit( parabola,  *zip(*scatterDataHull))

fit = list( start )
counter = 0
test = True
myOff = .1
while test:
    counter += 1
    ### "simple" fit with orthogonal distance
    fit, err = leastsq( residuals, x0=fit, args=( scatterDataHull, myOff ) )
    ### as points below the parabola are forbidden, we give those a high
    ### cost in the fitting. Starting with very high costs, however, 
    ### results in bad convergence. So we start with a rather low cost,
    ### check if some points are below and---if so---increase the cost.
    ### We then fit again with the previous result as starting point.
    ### (Note sometimes this fails, but I did not have time to
    ### investigate this cases. Moreover I hope that this is due 
    ### to my quite random points, while the points in the OP look 
    ### by for more regular.)
    #print "fit",fit 
    xList, yList = zip( *scatterDataHull )
    yFitList = [ parabola( x, *fit ) for x in xList ]
    signList = [ y - yTh  for y, yTh in zip( yList, yFitList ) ]
    #print counter, myOff
    #print signList
    if min( signList ) < 0:
        myOff *= 2
    else:
        test = False
    if counter > 50: # maxiter
        test = False
        warnings.warn( "Max iterations reached.", UserWarning )

startList = [ parabola( x, *start ) for x in xxList ]
fitList = [ parabola( x, *fit ) for x in xxList ]

fig = plt.figure()
ax = fig.add_subplot( 1, 1, 1 )
ax.plot( xxList, yyList , label='true parabula')
ax.plot( xxList, startList, label='simple fit' )
ax.plot( xxList, fitList, label='result')
ax.scatter( *zip( *scatterData ), s=60, label='full data')
ax.scatter( *zip( *scatterDataHull ), s=40, label='convex hull')
#ax.scatter( *zip( *scatterDataHullR ), s=20, label='lower hull' )
ax.legend( loc=0 )
ax.set_xlim( [ 0, 250 ] )
ax.set_ylim( [ 0, 250 ] )
#plt.show()
print(scatterDataHull)
