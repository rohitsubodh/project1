import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 
sns.set()
df = pd.read_csv('wine.data')
df.shape

df.columns=('class','Alcohol','Malic acid','Ash','Alcalinity of ash',
            'Magnesium','Total phenols','Flavanoids',
            'Nonflavanoid phenols','Proanthocyanins',
            'Color intensity','Hue',
            'OD280/OD315 of diluted wines','Proline')    
df.head();
X = df.iloc[:,1:]

y = df['class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_test_sc)

pca.components_

pca.explained_variance_ratio_

print(np.round(pca.explained_variance_ratio_,3))

pd.DataFrame(np.round(pca.components_,3),columns = X.columns).T

pca = PCA(n_components=None)
pca.fit(X_train_sc)
pca.fit_transform(X_train_sc)

print(np.round(pca.explained_variance_ratio_,3))

np.cumsum(np.round(pca.explained_variance_ratio_,3))

plt.plot(np.cumsum(np.round(pca.explained_variance_ratio_,3))*100)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()

res = pca.transform(X_train_sc)
ind_nam=["PCA_"+str(k) for k in range(0,len(res))]

df2 = pd.DataFrame(res, columns=df.columns[1:],
                  index=ind_nam)[0:4]
df2.T.sort_values(by='PCA_0')
df2.head()
