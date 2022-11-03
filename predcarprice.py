import pandas as pd
import seaborn as sns
import numpy as np
df=pd.read_csv("C:/ML projects/car_price/car data.csv")
df.shape
print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
##check missing values
df.isnull().sum()
df.describe()
final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
final_dataset.head()
final_dataset['Current Year']=2020
final_dataset.head()
final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']
final_dataset.head()
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.head()
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.head()
final_dataset=final_dataset.drop(['Current Year'],axis=1)
final_dataset.head()
final_dataset.corr()

sns.pairplot(final_dataset)
X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
X['Owner'].unique()
X.head()
y.head()
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.describe
y_train.describe
y_test.describe
from sklearn.ensemble import RandomForestRegressor
regressorRF=RandomForestRegressor()
regressorRF.fit(X_train, y_train)
y_predrf= regressorRF.predict(X_test)
from sklearn.metrics import r2_score
score2_rf = r2_score(y_test,y_predrf)
print("R-Square value with Random forest reg", score2_rf)
sns.distplot(y_test-y_predrf)
plt.scatter(y_test,y_predrf)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_predrf))
print('MSE:', metrics.mean_squared_error(y_test, y_predrf))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predrf)))
import pickle
file= open(r'C:\ML projects\car_price\proj_random_forest_regression_model.pkl', 'wb')
pickle.dump(regressorRF,file)
