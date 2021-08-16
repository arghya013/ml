import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("bigmarttest.csv")
#use required features
cdf = df[['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('bigmartsales.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('bigmartsales.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''