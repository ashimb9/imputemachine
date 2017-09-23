from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def MissForest(Ximp, categorical=None):
    """Imputer based on the missForest package in R.

    Please note that this is a work in progress."""

    mask = np.isnan(Ximp)
    missing_rows, missing_cols = np.where(mask)
    col_index = np.arange(Ximp.shape[1])
    continuous = np.setdiff1d(col_index, categorical)

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
    gamma_new = gamma_new_cat = 0
    gamma_old = gamma_old_cat = np.inf
    rf_reg = RandomForestRegressor(random_state=0, n_estimators=100)
    rf_clf = RandomForestClassifier(random_state=0, n_estimators=100)
    while (gamma_new < gamma_old or gamma_new_cat < gamma_old_cat) and iter < \
            max_iter:
        # added
        # 4. store previously imputed matrix
        Ximp_old = np.copy(Ximp)
        if iter != 0:
            gamma_old = gamma_new
            gamma_old_cat = gamma_new_cat
        # 5. loop
        for s in k:
            s_prime = np.setdiff1d(col_index, s)
            obs_rows = np.where(~mask[:, s])[0]
            mis_rows = np.where(mask[:, s])[0]
            yobs = Ximp[obs_rows, s]
            xobs = Ximp[obs_rows, :][:, s_prime]
            xmis = Ximp[mis_rows, :][:, s_prime]
            # 6. Fit a random forest
            if k in continuous:
                rf_reg.fit(X=xobs, y=yobs)
                # 7. predict ymis(s) using xmis(x)
                ymis = rf_reg.predict(xmis)
            else:
                rf_clf.fit(X=xobs, y=yobs)
                # 7. predict ymis(s) using xmis(x)
                ymis = rf_clf.predict(xmis)

            Ximp[mis_rows, s] = ymis
            # 8. update imputed matrix using predicted matrix ymis(s)
        # 9. Update gamma
        gamma_new = np.sum(
            (Ximp_old[:, continuous] - Ximp[:, continuous]) ** 2) / np.sum(
            (Ximp[:, continuous]) ** 2)
        gamma_new_cat = np.sum(
            Ximp_old[:, categorical] != Ximp[:, categorical]) / np.sum(
            mask[:, categorical])

        print("Iteration:", iter)
        iter += 1
    return Ximp_old
