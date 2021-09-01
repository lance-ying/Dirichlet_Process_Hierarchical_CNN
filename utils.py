
from sklearn import mixture
import numpy as np
import torch

def clustering(X, n_components=2, method="random"):
    DPGMM = mixture.BayesianGaussianMixture(n_components=n_components, 
                                                max_iter=10000,
                                                n_init=5,
                                                tol=1e-5,
                                                init_params=method, 
                                                random_state=42,
                                                weight_concentration_prior_type='dirichlet_process',
                                                weight_concentration_prior=1)
    DPGMM.fit(X)
    return DPGMM

def predict(clf, DPMM, x, mfb):

    pred=torch.zeros((1,3))

    for i, model in enumerate(clf):
        model=clf[i].cpu()
        soft_label=torch.nn.Softmax(dim=1)(model(torch.tensor(mfb.reshape(1,mfb.shape[0],mfb.shape[1]))))
        p=DPMM.predict_proba(x.reshape(1,-1))[0][i]*soft_label

        pred+=p

    return np.argmax(pred.detach().numpy())

