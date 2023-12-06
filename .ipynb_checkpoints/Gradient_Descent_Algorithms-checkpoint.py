import numpy as np

def zscoreNormalize(m, n):
    featNorm = np.zeros((m,n))
    featMean = np.mean(feat, axis=0)
    featStd = np.std(feat, axis=0)
    for i in range(m):
        for j in range(n):
            featNorm[i,j] = (feat[i,j] - featMean[j])/featStd[j]

    return featNorm

def gradDescentLinear(feat, targ, alpha, iterations):
    
    # input format: 
    
    # multiple linear regression for features x1, x2, .. , xn (each with m values):
    # feat = [[x1(1), x2(1), x3(1), ... xn(1)],
    #         [x1(2), x2(2), x3(2), ... xn(2)],
    #         [....., ....., ....., ... xn(m)]
    
    # simple linear regression for feature with m values:
    # feat = [x(1), x(2), x(3), ... x(m)]

    # target array with m values:
    # targ = [y(1), y(2), y(3), ... y(m)]

    # calculating cost function and derivatives
    def costLinear(m, n, w, featNorm, b, targ):
        cost = 0
        dw = np.zeros(n)
        db = 0
        
        for i in range(m):
            error = np.dot(w, featNorm[i]) + b - targ[i]
            cost += error ** 2
            dw += error * featNorm[i]
            db += error
            
        cost = cost / (2 * m)
        db = db / m
        dw = dw / m

        return cost, dw, db

    if feat.ndim == 1:
        feat = feat[:, np.newaxis]  # transposes array for simple linear regression

    
    m = len(targ) # length of each feature array
    n = feat.shape[1] # number of features

    # Normalize features for better convergence

    featNorm = zscoreNormalize(m, n)

    # intiializing variables
    w = np.zeros(n) 
    b = 0 
    
    #initializing lists
    costHistory = np.zeros(iterations)
    wHistory = np.zeros((iterations, n))
    bHistory = np.zeros(iterations)
    
    for i in range(iterations):
        cost, dw, db = costLinear(m, n, w, featNorm, b, targ)
        w -= alpha * dw
        b -= alpha * db
        wHistory[i,:] = w
        bHistory[i] = b
        costHistory[i] = cost
    
    return w, b, costHistory, wHistory, bHistory, featNorm

def gradDescentPoly(X, Y, n, alpha, iterations):
    
    # input format: 
    # X = [x(1), x(2), x(3), ... x(m)] (single feature array of length m)
    # Y = [y(1), y(2), y(3), ... y(m)] (target array of length m)
    # n = degree of desired polynomial


    X = X[:, np.newaxis]
    X = np.hstack((X, np.zeros((X.shape[0], n-1))))

    for i in range(1, n):
        X[:, i] = X[:,0] ** (i + 1)
        
    w, b, costHistory, wHistory, bHistory, XNorm = gradDescentLinear(X, Y, alpha, iterations)
    return w, b, costHistory, wHistory, bHistory, XNorm