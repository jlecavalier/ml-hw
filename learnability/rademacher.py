from random import randint, seed
from collections import defaultdict
import math

from itertools import combinations

from numpy import array
from numpy.linalg import norm

from bst import BST

kSIMPLE_DATA = [(1, 1), (2, 2), (3, 0), (4, 2)]

class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

        sigma = coin_tosses(len(data))

        bigsum = 0.0
        for i in range(len(data)):
        	bigsum += float(labels[i]) * float(classify_to_int(self.classify(data[i])))

        return (bigsum/float(len(data)))


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """
    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True

def classify_to_int(value):
	if value:
		return 1
	else:
		return -1

def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # TODO: Complete this function

    yielded = []

    for ii in dataset:
      	r = math.hypot(ii[0],ii[1])
      	theta = math.acos(ii[0]/r)
      	x = math.cos(theta-math.pi/2)
      	y = math.sin(theta-math.pi/2)
      	if ((x,y) not in yielded):
      		yielded.append((x,y))

    for ret in yielded:
    	yield OriginPlaneHypothesis(ret[0], ret[1])
    	yield OriginPlaneHypothesis(-ret[0], -ret[1])

def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # Complete this for extra credit
    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """
    # Make a rectangle that contains no data points
    yield AxisAlignedRectangle(float('inf'),float('inf'),float('inf'),float('inf'))
    to_yield = []
    # Add all rectangles containing just one point
    for ii in dataset:
        to_yield.append(AxisAlignedRectangle(ii[0],ii[1],ii[0],ii[1]))

    for n in range(2,len(dataset)+1):
        potentials = combinations(dataset,n)
        for ii in potentials:
            p_x = []
            p_y = []
            for jj in ii:
                p_x.append(jj[0])
                p_y.append(jj[1])
            rect = AxisAlignedRectangle(min(p_x),min(p_y),max(p_x),max(p_y))
            pcount = 0
            for kk in dataset:
                if rect.classify(kk):
                    pcount = pcount + 1
            if pcount == n:
                if rect not in to_yield:
                    to_yield.append(rect)
    
    for r in to_yield:
        yield r


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """
    maxes=[]
    for i in range(num_samples):
    	correlations = []
    	if i == 0:
    		sigma = coin_tosses(len(dataset),random_seed)
    	else:
    		sigma = coin_tosses(len(dataset))
    	for kk in hypothesis_generator(dataset):
    		correlations.append(kk.correlation(dataset,sigma))
    	maxes.append(max(correlations))
    return sum(maxes)/len(maxes)

if __name__ == "__main__":
	print("Rademacher correlation of constant classifier %f" % rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
	print("Rademacher correlation of rectangle classifier %f" % rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
	print("Rademacher correlation of plane classifier %f" % rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
