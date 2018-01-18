import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def compute_mse(X,Y):
	return np.sqrt(np.mean(np.square(X-Y)))

def ohe(X):
	u_x = list(set(X))
	print(u_x)
	c = len(u_x)
	r = X.shape[0]
	oh = np.zeros([r,c])
	for i in range(c):
		for j in range(r):
			if X[j] == u_x[i]:
				oh[j,i] = 1
			else:
				oh[j,i] = 0
	return oh

 	
mask_v=[]
mask_tr=[]
mask_ts=[]
'''
discrete_variables = ["MS SubClass", ’MS Zoning’, ’Street’,
’Alley’, ’Lot Shape’, ’Land Contour’,
’Utilities’, ’Lot Config’, ’Land Slope’,
’Neighborhood’, ’Condition 1’, ’Condition 2’,
’Bldg Type’, ’House Style’, ’Overall Qual’,
’Overall Cond’, ’Roof Style’, ’Roof Matl’,
’Exterior 1st’, ’Exterior 2nd’, ’Mas Vnr Type’,
’Exter Qual’, ’Exter Cond’, ’Foundation’,
’Bsmt Qual’, ’Bsmt Cond’, ’Bsmt Exposure’,
’BsmtFin Type 1’, ’Heating’, ’Heating QC’,
’Central Air’, ’Electrical’, ’Bsmt Full Bath’,
’Bsmt Half Bath’, ’Full Bath’, ’Half Bath’,
’Bedroom AbvGr’, ’Kitchen AbvGr’, ’Kitchen Qual’,
’TotRms AbvGrd’, ’Functional’, ’Fireplaces’,
’Fireplace Qu’, ’Garage Type’, ’Garage Cars’,
’Garage Qual’, ’Garage Cond’, ’Paved Drive’,
’Pool QC’, ’Fence’, ’Sale Type’, ’Sale Condition’]

'''

url='https://ww2.amstat.org/publications/jse/v19n3/decock/AmesHousing.txt'
data = np.genfromtxt(url, delimiter="\t", dtype=None)
header_list = data[0,:]
print(header_list)
data = np.delete(data, 0, 0)

len_validation=0
len_test=0
len_train =0

#lengths of validation set 
for i in range(data.shape[0]):
	if int(data[i,0])%5 ==3:
		len_validation+=1
		mask_v.append(i)
	elif int(data[i,0])%5 ==4:
		len_test+=1
		mask_ts.append(i)
	else:
		len_train+=1
		mask_tr.append(i)

print("length of validation set =",len_validation)
print("length of test set =",len_test)
print("length of train set =",len_train)
validation=np.zeros([len_validation,data.shape[1]])
train=np.zeros([len_train,data.shape[1]])
test=np.zeros([len_test,data.shape[1]])

validation = data[mask_v]
train= data[mask_tr]
test= data[mask_ts]
print(validation.shape)
print(test.shape)
print(train.shape)

#Extracting sale price
sale_train = train[:,data.shape[1]-1]
gr_area_train = train[:,47]
sale_train = sale_train.astype(float)
gr_area_train = gr_area_train.astype(float)
#validation
sale_val = validation[:,data.shape[1]-1]
gr_area_val = validation[:,47]
sale_val = sale_val.astype(float)
gr_area_val = gr_area_val.astype(float)

print("Sales Data shape = ",sale_train)
print("gr_area data shape = ",gr_area_train)
#scatter Plot in python

#plt.scatter(gr_area_train,sale_train)
#plt.show()

# Regression
a =1/(np.dot(gr_area_train,gr_area_train))
b= np.dot(gr_area_train,sale_train)
beta = a*b
print("beta = ", a*b)
y_pred = np.zeros_like(gr_area_val)
#Prediction of validation data
y_pred = gr_area_val*beta

print( "MSE =", compute_mse(y_pred,sale_val))

#Extracting the indices for all he discreet variable
index =[j for j, e in enumerate(header_list) if e in discrete_variables]
model_data = data

# One hot encoding
#1.Alley
#alley_oh = ohe(data[:,7])
#for i in range(len(index)):
#	x = ohe(data[:,index[i]])	
