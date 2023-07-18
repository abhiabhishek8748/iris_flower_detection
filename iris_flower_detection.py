import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
dataset = load_iris()
print(dataset.DESCR)
x = dataset.data
y = dataset.target
y
x
plt.plot(x[:, 0][y == 0]* x[:, 1][y == 0],x[:, 1][y == 0]* x[:, 2][y == 0],'r.',label='Setosa')
plt.plot(x[:, 0][y == 1]* x[:, 1][y == 1],x[:, 1][y == 1]* x[:, 2][y == 1],'g.',label='Versicolor')
plt.plot(x[:, 0][y == 2]* x[:, 1][y == 2],x[:, 1][y == 2]* x[:, 2][y == 2],'b.',label='Virginica')
plt.legend()
plt.show
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y)
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier()
rfm.fit(x_train,y_train)
rfm.score(x_test, y_test)
rfm.score(x,y)