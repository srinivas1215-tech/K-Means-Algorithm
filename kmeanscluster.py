import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

#Encoding Module - for convert string datapoints to numbers
def encoding(X):  
    for col in X.columns:
        if X[col].dtype == 'object': # Encode only columns with object dtype
            X[col] = X[col].astype('category').cat.codes
    return X

#normalizing module for distant data points
def normalize(X):  
    for col in X.columns:
        min_val = np.min(X[col])
        max_val = np.max(X[col])

        nor_col = (X[col] - min_val) / (max_val - min_val)
        X[col] = nor_col
    return X

#Euclidean distance calculation
def eucli_dist(x,y):
    a=sum((x - y)**2)
    a=math.sqrt(a)
    return a

#Initialize Centroids
def centroid_initial(X,y,K):
    # Generate K random unique indices
    rand_indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[rand_indices]
    lab = y[rand_indices]    
    lab.astype(int)
    return centroids,lab

#Cluster assigning
def cluster_assign(X, K, centroids):  # clustering-part
    cl = [[] for _ in range(K)]
    for i, val in enumerate(X):
        dis = []
        for cent in centroids:
            s = eucli_dist(val, cent)
            dis.append(s)
        # Ensure distances are computed
        if len(dis) == 0:
            raise ValueError("Distance array is empty")
        index = np.argmin(dis)
        if index < len(cl):
            cl[index].append(i)
    return cl

#predicted label assigning for cluster
def label_assign(lab,cl):
    pl=[[]for _ in range(len(lab))]
    for i,c in enumerate(cl):
        for a in c:
            pl[lab[i]].append(a)
    return pl

#Update centroids 
def updatecent(K, cl, X):
    newcent = np.zeros((K, X.shape[1]))  # Initialize new centroids
    for i in range(K):
        if len(cl[i]) > 0:
            newcent[i] = X[cl[i]].mean(axis=0)  # Mean of points in the cluster
            
    return newcent   

   
def find_kmeans(X,y,K):       
    centroids,lab = centroid_initial(X,y,K)
    
    print("\n------------------------------------")
    print("LABELS \t \t CENTROIDS")
    print("------------------------------------")
    for i,cent in enumerate(centroids):
        print(lab[i],end='')
        for c in cent:
            print("\t\t",c)
        print("----------------------------------")
    while(True):
        cl=cluster_assign(X,K,centroids)
        newcent=updatecent(K,cl,X)
        if np.all(centroids == newcent):
            break
        centroids=newcent
    pl = label_assign(lab,cl)        
    return centroids,cl,lab,pl
    
            
            
if __name__ == "__main__":
    
    df = pd.read_csv("./iris.csv")
    #df = df.drop('Id',axis=1)
    col = df.columns.tolist()
    l=col.pop()
    df = df.dropna()
    df = encoding(df)
    #print(df)

    #select the Feature
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    X=normalize(X)
    print(X)
    print(y)
    

    train_x, test_x,train_y,test_y= tts(X,y,test_size=0.2, random_state=42)    
    
    train_x = np.array(train_x)
    test_x  = np.array(test_x)

    train_y = np.array(train_y)
    test_y  = np.array(test_y)

    K=int(input("\nEnter value for K : "))
    centroids,cl,labels,pl= find_kmeans(train_x,train_y,K)
    

    ft=pd.DataFrame(test_x,columns=col)
    la=pd.DataFrame(test_y,columns=[l])
    
    df = pd.concat([ft,la],axis=1)
    clust = cluster_assign(test_x,K,centroids)
    test = label_assign(labels,clust)
    print("\n-------------------------------------------------------------")
    print("CLUSTERS  \t  FINAL CENTROIDS ")
    print("-------------------------------------------------------------")
    
    for i,cent in enumerate(centroids):
        print(len(clust[i]),end='')
        for c in cent:
            print("\t\t ",c)
        print("-------------------------------------------------------------")
        print("\tPREDICTED LABEL : ",labels[i])
        print("-------------------------------------------------------------") 
    
    for i,m in enumerate(test):
        for j in m:
            for a in range(len(test_x)):
                if j==a:
                    df.loc[j,"Predicted"]=i
                    break
    df.to_csv("./testoutput.csv")
    y_pred=df.iloc[:,-1].values
    accuracy = accuracy_score(test_y,y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")




