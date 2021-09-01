import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import * 
from utils import *
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import random
import argparse

###setting random seeds
def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



def main():

    parser = argparse.ArgumentParser(description='NHNN Version')
    parser.add_argument('version', type=str, help='input NHNN version: choose from FC+Conv or FC')
    version=parser.parse_args().version

    if version not in ["FC+Conv", "FC"]:
        print("invalid input")
        return 0

    os.mkdir("./checkpt")
    print("loading data")

    X=np.load("./IEMOCAP.npy")

    df=pd.read_csv("./data.csv")

    Y=df["valence"].to_numpy()
    features=pd.read_csv("./features.csv")
    X_feat=StandardScaler().fit_transform(features.drop(["name"],axis=1))

    folds=len(df["sub_id"].unique())

    print("data loaded")

    DPMM=clustering(X_feat,2)

    group_kfold = GroupKFold(n_splits=folds)
 
    UAR=np.zeros((2,folds))

    count=0

    ###training with LOSO

    for train_idx, test_idx in tqdm(group_kfold.split(X, Y, df["sub_id"])):
        X_TR=np.take(X,train_idx,axis=0)
        Y_TR=np.take(Y,train_idx,axis=0)

        X_test=np.take(X,test_idx,axis=0)
        Y_test=np.take(Y,test_idx,axis=0)

        feat_train=np.take(X_feat,train_idx,axis=0)
        X_feat_test=np.take(X_feat,test_idx,axis=0)


        print("fold:",count)

        ###repeat experiments with different random seeds
        for trial in range(2):
            seed_everything(42+trial)


            path=f"./checkpt/fold_{count}/"

            clf=[]
            
            for i in range(len(DPMM.weights_)):
                cluster_index=np.array(np.where(DPMM.predict(feat_train)==i)[0])
                Tr_idx=cluster_index[:int(0.8*len(cluster_index))]
                Val_idx=cluster_index[int(0.8*len(cluster_index)):]

                X_train=np.take(X_TR,Tr_idx,axis=0)
                X_val=np.take(X_TR,Val_idx,axis=0)
                Y_train=np.take(Y_TR,Tr_idx,axis=0)
                Y_val=np.take(Y_TR,Val_idx,axis=0)

                train_data=TensorDataset(torch.FloatTensor(X_train),torch.LongTensor(Y_train))
                val_data=TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(Y_val))
                train_loader=DataLoader(train_data, batch_size=64, shuffle=True)
                val_loader=DataLoader(val_data, batch_size=64, shuffle=True)


                patience=5

                model = NHNN()

                model.load_state_dict(torch.load(path+"{}.checkpoint".format(trial)))

                if version=="FC+Conv":
                    ###freezing FC and one Conv layer
                    for layer in model.cnn_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                        break
                else:

                    ###freezing only FC layers
                    for param in model.cnn_layers.parameters():
                        param.requires_grad = False

                best_model=None
                best_loss=np.inf

                cur_patience=patience

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                if torch.cuda.is_available():
                    model = model.cuda()
                    criterion = criterion.cuda()

                
            #training
                while cur_patience!=0 and model.epoch<50:
                    model.train()
                    model.fit(optimizer, criterion, train_loader,val_loader)

                    if model.loss<=best_loss:
                        best_model=model.state_dict()
                        best_loss=model.loss
                        cur_patience=patience
                    else:
                        cur_patience-=1

                    model.epoch+=1

                with torch.no_grad():
                    model= NHNN()
                    model.load_state_dict(best_model)
                    clf.append(model)

            pred=[]
            for feat, x in zip(X_feat_test,X_test):
                pred.append(predict(clf, DPMM, feat, x))
            
            perf=metrics.recall_score(Y_test,np.array(pred), average="macro")
            print(count,trial,perf)

            UAR[trial,count]=perf

        count+=1


    ###saving results
    np.save("./IEMOCAP_NHNN_fc.npy",UAR)

    print("UAR:",UAR.mean())
        

if __name__ == "__main__": 


    main()

            
        
