import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from DPHNN import * 
from utils import *
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler
import random

torch.manual_seed(10)
torch.cuda.manual_seed(10)
random.seed(10)


os.chdir("/home/lancelcy/PRIORI")




def main():
    # parser = argparse.ArgumentParser(description='trial.')
    # parser.add_argument('trial', type=str, help='input trial no.')
    # t=parser.parse_args().trial
    print("loading data")

    # df=pd.read_csv("/home/lancelcy/PRIORI/metadata/R21_f.csv")
    df=pd.read_csv("/home/lancelcy/PRIORI/metadata/V1_v2.csv")
    X=np.load("/home/lancelcy/PRIORI/data/V1_norm.npy")
    X=X[:,:,:6000]
    # idx=np.load("/home/lancelcy/PRIORI/metadata/R21_idx.npy")
    # X=np.take(X,idx, axis=0)
    Y=df["valence"].to_numpy()
    features=pd.read_csv("/z/lancelcy/V1/features_norm.csv")
    X_feat=StandardScaler().fit_transform(features.drop(["name"],axis=1))
    print(X.shape)
    print(Y.shape)

    folds=len(df["subject_id"].unique())
    # folds=19
    print("data loaded")

    group_kfold = GroupKFold(n_splits=folds)
    # UAR=np.zeros((20, folds))
    DPMM=clustering(X_feat,2)
    # DPMM=pickle.load(open("/home/lancelcy/PRIORI/train_script/CNN/DPMM_random.pk","rb"))
    # for trial in tqdm(range(20)):
    UAR=np.zeros(19)
    for trial in [1]:
        # path="/home/lancelcy/PRIORI/checkpt/R21_CNN/LOO_norm/LOO_{}/".format(trial)
        path="/home/lancelcy/PRIORI/checkpt/CNN_models/valence/full/"
        count=0

        for train_idx, test_idx in tqdm(group_kfold.split(X, Y, df["subject_id"])):
            X_TR=np.take(X,train_idx,axis=0)
            Y_TR=np.take(Y,train_idx,axis=0)

            X_test=np.take(X,test_idx,axis=0)
            Y_test=np.take(Y,test_idx,axis=0)

            feat_train=features.loc[train_idx]
            feat_test=features.loc[test_idx]

            X_feat_test=StandardScaler().fit_transform(feat_test.drop(["name"],axis=1))

            clf=[]
            
            for i in range(len(DPMM.weights_)):
                cluster_index=np.array(np.where(DPMM.predict(StandardScaler().fit_transform(feat_train.drop(["name"],axis=1)))==i)[0])
                np.random.shuffle(cluster_index)
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


        # training the model
        # for i in tqdm(range(5)):
            #initializing
                patience=5
                model = Net(patience)
                model.load_state_dict(torch.load(path+"{}.checkpoint".format(count)))

                for layer in model.cnn_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                    break

                # for param in model.cnn_layers.parameters():
                    # param.requires_grad = False
                best_model=None
                best_loss=np.inf

                cur_patience=patience

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-6)
                if torch.cuda.is_available():
                    model = model.cuda(1)
                    criterion = criterion.cuda(1)

                
            #training
                while cur_patience!=0 and model.epoch<80:
                    model.train()
                    model.fit(optimizer, criterion, train_loader,val_loader)

                    if model.loss<=best_loss:
                        # print(new_val_loss)
                        best_model=model.state_dict()
                        # print(best_model)
                        best_loss=model.loss
                        cur_patience=patience
                    else:
                        cur_patience-=1

                    model.epoch+=1

                with torch.no_grad():
                    model= Net()
                    model.load_state_dict(best_model)
                    clf.append(model)

            pred=[]
            for feat, x in zip(X_feat_test,X_test):
                pred.append(predict(clf, DPMM, feat, x))
            
            # Y_out=model(torch.from_numpy(X_test)).numpy()
            # Y_pred=np.zeros(Y_test.shape[0])
            # for j, pred in enumerate(Y_out):
            #     Y_pred[j]=np.argmax(pred)
            perf=metrics.recall_score(Y_test,np.array(pred), average="macro")
            print(trial,count,perf)

            UAR[count]=perf
            # UAR[trial,count]=perf
            # print(i)
            # filename=os.path.join(f"/home/lancelcy/PRIORI/checkpt/R21_CNN",f"{i}.checkpoint")
            # print(filename)
            # with torch.no_grad():
                # torch.save(best_model, filename)
            count+=1

            np.save("/home/lancelcy/PRIORI/test_result/V1_val_DPHNN_cnn.npy",UAR)
        # print(np.array(UAR).mean())
        print(UAR.mean())
    np.save("/home/lancelcy/PRIORI/test_result/V1_val_DPHNN_cnn.npy",UAR)
    # print(UAR.mean(axis=1))
        

if __name__ == "__main__": 

    main()

            
        