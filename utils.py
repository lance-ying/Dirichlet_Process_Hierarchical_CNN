from sklearn import mixture
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def clustering(X, n_components=2, method="kmeans"):
    DPGMM = mixture.BayesianGaussianMixture(n_components=n_components, 
                                                max_iter=10000,
                                                n_init=5,
                                                tol=1e-5,
                                                init_params=method, 
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=1/10)
    DPGMM.fit(X)
    return DPGMM

def predict(clf, DPMM, x, mfb):
    # print(DPMM.predict(x.reshape(1,-1)))
    model=clf[DPMM.predict(x.reshape(1,-1))[0]].cpu()
    # print(mfb.shape)

    pred=model(torch.tensor(mfb.reshape(1,mfb.shape[0],mfb.shape[1])))
    return np.argmax(pred.detach().numpy())
