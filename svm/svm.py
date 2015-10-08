from numpy import array, zeros

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))
    for i in range(len(alpha)):
    	# a_i * y_i
    	factor = alpha[i] * y[i]
    	for j in range(len(w)):
    		w[j] += factor * x[i][j]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()
    for i in range(len(x)):
    	p = w.dot(array(x[i])) + b
    	# x[i] is a support vector on the positive side
    	if p <= 1 + tolerance and p >= 1 - tolerance:
    		support.add(i)
    	# x[i] is a support vector on the negative side
    	if p >= -1 - tolerance and p <= -1 + tolerance:
    		support.add(i)

    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()
    for i in range(len(x)):
    	# If p >= 1, then p should be in the positive class.
    	# If p <= 1, then p should be in the negative class.
    	p = w.dot(array(x[i])) + b
    	# x[i] was classified as positive, but it's really negative.
    	if p >= 1 and y[i] < 0:
    		slack.add(i)
    	# x[i] was classified as negative, but it's really positive.
    	if p <= -1 and y[i] > 0:
    		slack.add(i)
    return slack
