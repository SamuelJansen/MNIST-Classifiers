from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


mnist = fetch_openml('mnist_784', cache=False)
X_dataset = mnist.data.astype('float32')
y_dataset = mnist.target.astype('int64')
X, Xt, y, yt = train_test_split(X_dataset, y_dataset, test_size=1/7, ) #random_state=42)
#- verifying if the sum of the disjointed parts makes the MNIST data set again
assert(X.shape[0] + Xt.shape[0] == mnist.data.shape[0]) 

###############################################################################
#- Plot functions - Made for tests and debugging
###############################################################################
def plot_sample(X, y, SubPlot):
    """Plot a image and it's label.
    In Colaboratory, if this function is called multiple times in a single run,
    the screen will only keeps the last plot call"""
    plt.subplot(1,1,1)
    plt.imshow(X.reshape(28, 28),cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.title(y)
    return SubPlot+1
###plot_sample(X[t], y[t])

def plot_samples(X, y, t, t_end, SubPlot):
    """Plot a k images images and it's label.
    In Colaboratory, if this function is called multiple times in a single run,
    the screen will only keeps the last k plots call"""
    for i in range(t_end-t) :
        plt.subplot(1,SubPlot+t_end-t-1,SubPlot + i)
        plt.imshow(X[t+i].reshape(28, 28),cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.title('y_ind={}\nlable={}'.format(t+i,y[t+i]))
    return SubPlot+t_end-t
###- plot_samples(X, y, t, t_end)
###- It will plot X[t], X[t+1], X[t+2], ..., X[t+t_end] 

SubPlot = 1

###############################################################################
#- Calculating one center point for each number within 0 and 9
#- from the MNIST trainning dataset
###############################################################################0
#- Spliting MNIST tranning set into 10 disjointed subsets, i.e., SubSets.
#- One for each number within 0 and 9
SubSets = dict()
for cl in range(k) :
    SubSets[cl] = []
for n in range(len(y)) :
    SubSets[y[n]].append(X[n])
#print( 'The size of subset is {}'.format( len(SubSets) ) )

#- making the center point f each class (i.e., cp of each number within 0 and 9)
#- and initializing it with zeros
CPclasses = []
for cl in range(k) :
    CPclasses.append( np.zeros((image_size_DS**2,),dtype=int) )

#- making the center point f each class
for n in range(len(y)) :
    CPclasses[y[n]] = CPclasses[y[n]] + X[n]

#- Calculating how many elements (i.e., points) each subset has
Classes_size = []        #- the number of elements in each subset
for cl in range(k) :
    Classes_size.append( len(SubSets[cl]) )
    CPclasses[cl] = CPclasses[cl] / Classes_size[cl] # CPclasses[cl] //= Classes_size[cl] #- Each feature of each class CP as a "integer"
#print( 'The ammount of elements in each subset is {}'.format(Classes_size) )

#- The imediate following code lines prints the center point of the 0 class
#cl_here=0
#plot_sample(CPclasses[cl_here], 'CPclass[{}]'.format(cl_here))
    
###############################################################################
#- Prediction based on a hash of the difference between
#- CPclasses and each test data points - Central Point of each Class
###############################################################################
def CPclassesPrediction(CPclasses, Xt, k, t):
    Prediction = 0          #- prediction
    summ = 0

    for cl in range(k) :
        Hash = CPclasses[cl]-Xt[t]
        np.warnings.filterwarnings('ignore') #- To ignore any "division by zero" warnings
        Hash = Hash / abs(Hash) # Hash //= abs(Hash)
    
    #- The imediate following code lines prints a hash of (CPClasses-Xt[0])
    ##cl_here=9
    #plot_sample(Hash, 'Hash[{}]'.format(cl_here))

        summ_cl = sum(Hash)
        if summ_cl>summ :
            Prediction = cl     #- update of the prediction
            summ = summ_cl
    return Prediction

###############################################################################
#- Calculates how many right predictions were made
###############################################################################
def RIGHT_PREDICTIONS(byas, CPclasses, CP, DistFromCP, Predictions_Cnn, Xt, yt, k) :
    CPclasses += np.asarray(byas) #- Byas
    right_predictions = 0

    for t in range(len(yt)) :
        #- Predictions
        Prediction_CPclasses = CPclassesPrediction(CPclasses, Xt, k, t)

        if Prediction_CPclasses==yt[t] :
            right_predictions+=1
            
    return right_predictions
    
###############################################################################
#- Optimum Byas for CPclasses - Central Point of each Class
###############################################################################
### uncommenting and running it takes a lot of time to compute
""" #- optimum byas = 224 
SamplesAmount = Xt.shape[0] # 100 #
for byas in range(100,300) :
    right_predictions = RIGHT_PREDICTIONS(byas, CPclasses, CP, DistFromCP, Predictions_Cnn, Xt[:SamplesAmount], yt[:SamplesAmount], k)
    print('      For byas = {}, it made right predictions {}% of the times'.format(byas, 100*right_predictions/SamplesAmount) )
#"""

###############################################################################
#- Classifying Xt data using CPclasses
###############################################################################
SamplesAmount = Xt.shape[0] # 100 #
byas = 224 #- CP search parameter #- optimum byas in colaboratory = 224

right_predictions = RIGHT_PREDICTIONS(byas, CPclasses, CP, DistFromCP, Predictions_Cnn, Xt[:SamplesAmount], yt[:SamplesAmount], k) # Xt, yt, k) #

print( 'CPclasses classification accuracy {}%'.format(100*right_predictions/SamplesAmount) )
