from celer import ElasticNet
from libsvmdata import fetch_libsvm
import numpy as np
from numpy.linalg import norm


dataset = 'rcv1.binary'
reg = 0.1

X, y = fetch_libsvm(dataset)
alpha_max = norm(X.T @ y, ord=np.inf) / X.shape[0]

params = {'l1_ratio': 0.5, 'alpha': reg*alpha_max,
          'fit_intercept': True, 'max_iter': 10000, 'tol': 0., 'verbose': 2}

model = ElasticNet(**params)
model.fit(X, y)


print(model.coef_)
