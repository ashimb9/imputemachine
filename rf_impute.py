from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def RFImputer(Ximp):
    """Based on the missForest package in R.

    Currently only supports continuous features. Categorical features will
    be supported soon. Also, please note that this is a work in progress."""

    mask = np.isnan(Ximp)
    missing_rows, missing_cols = np.where(mask)

    # MissForest Algorithm
    # 1. Make initial guess for missing values
    col_means = np.nanmean(Ximp, axis=0)
    Ximp[(missing_rows, missing_cols)] = np.take(col_means, missing_cols)

    # 2. k <- vector of sorted indices of columns in X
    col_missing_count = mask.sum(axis=0)
    k = np.argsort(col_missing_count)

    # 3. While not gamma_new < gamma_old and iter < max_iter do:
    iter = 0
    max_iter = 100
    gamma_new = 0
    gamma_old = np.inf
    col_index = np.arange(Ximp.shape[1])
    model_rf = RandomForestRegressor(random_state=0, n_estimators=1000)
    # TODO: Update while condition for categorical vars
    while gamma_new < gamma_old and iter < max_iter:
        # added
        # 4. store previously imputed matrix
        Ximp_old = np.copy(Ximp)
        if iter != 0:
            gamma_old = gamma_new
        # 5. loop
        for s in k:
            s_prime = np.delete(col_index, s)
            obs_rows = np.where(~mask[:, s])[0]
            mis_rows = np.where(mask[:, s])[0]
            yobs = Ximp[obs_rows, s]
            xobs = Ximp[np.ix_(obs_rows, s_prime)]
            xmis = Ximp[np.ix_(mis_rows, s_prime)]
            # 6. Fit a random forest
            model_rf.fit(X=xobs, y=yobs)
            # 7. predict ymis(s) using xmis(x)
            ymis = model_rf.predict(xmis)
            Ximp[mis_rows, s] = ymis
            # 8. update imputed matrix using predicted matrix ymis(s)
        # 9. Update gamma
        gamma_new = np.sum((Ximp_old - Ximp) ** 2) / np.sum(
            (Ximp) ** 2)
        print("Iteration:", iter)
        iter += 1
    return Ximp_old