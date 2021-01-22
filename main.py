"""
# PCA, Regression, and SVM
# Hanzallah Azim Burney
"""

# Import Libraries
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat 
from sklearn import svm

################################ Principal Component Analysis Functions ################################

# PCA using SVD 
def SVD(matrix):
  U, s, V_t = np.linalg.svd(matrix, full_matrices=False)
  return s

def PVE(sigma):
  pve = [sigma[i]/np.sum(sigma) * 100 for i in range(100)]
  return pve

################################ Linear Regression Functions ################################

# performance metrics
def SSE(actual, predicted):
  squared_errors = (actual - predicted)**2
  return np.sum(squared_errors)

def SST(actual, mean):
  squares_total = (actual - mean)**2
  return np.sum(squares_total)

def MAE(actual, predicted):
   absolute_errors = np.absolute(actual - predicted)
   return np.sum(absolute_errors)

def MAPE(actual, predicted):
  absolute_percentage_errors = np.absolute((actual - predicted)/actual)
  return np.sum(absolute_percentage_errors)/actual.shape[0]

# creating the linear regression models for 5-fold cross validation
def k_fold_validate(data, batch_size = 100, k = 5):
  r, c = data.shape

  # randomly shuffle data
  np.random.shuffle(data) 

  # create 5 batches
  folds = [data[i:i+batch_size] for i in range(0,r,batch_size)]

  train = []
  test = []

  for i in range(0,len(folds)):
    begin = []
    end = []
    if (i > 0):
      begin = np.concatenate(folds[:i])
    if (i < len(folds)-1):
      end = np.concatenate((folds[i+1:]))
    
    if (len(begin) > 0 and len(end) > 0):
      train.append(np.concatenate((begin,end)))
    elif (len(begin) > 0):
      train.append(begin)
    elif (len(end) > 0):
      train.append(end)
    test.append(folds[i])
  
  return train,test

def train_linear(train):
  weights = []
  for train_data in train:
    x = train_data[:, :-1]
    y = train_data[:, -1].reshape(-1,1)

    # linear regression
    cur_weights = np.dot(np.linalg.inv(np.dot(x.T,x)), np.dot(x.T,y))
    weights.append(cur_weights)

  return weights

# linear regression with lasso regularization
# lr 0.01 and 1
def lasso_gradient_descent(data, penalty=1, lr = 0.01,iterations=1000000):
  train, test = k_fold_validate(data)
  weights = []
  
  for train_data in train:
    x = train_data[:, :-1]
    y = train_data[:, -1].reshape(-1,1)
    r,c = x.shape
    cur_weights = np.zeros(c).reshape(-1,1)

    for i in range(iterations):
      y_pred = np.dot(x,cur_weights)

      # calculate gradient
      gradient = np.dot(x.T, y_pred-y) + (penalty*np.sign(cur_weights))

      # update weights
      cur_weights -= 1/r * lr*gradient

    weights.append(cur_weights)

  return weights, test
    
def predict(weights, test):
  # predict for each test set
  fold_perf = []
  for i in range(len(test)):
    y_pred = np.dot(test[i][:,:-1], weights[i])
    
    # Caluclate performance of each fold
    print(f'Fold {i}')
    
    # test labels
    test_labels = test[i][:,7].reshape(-1,1) 

    # SSE
    sse = SSE(test_labels,y_pred.reshape(-1,1))

    # SST
    y_mean = np.mean(test[i])
    sst = SST(test_labels, y_mean)

    # R^2
    r2 = 1 - sse/sst
    print(f'R^2 error: {r2}')

    # MSE
    mse = sse / test_labels.shape[0]
    print(f'Mean Squared Error: {mse}')

    # MAE
    mae = MAE(test_labels,y_pred.reshape(-1,1))
    print(f'Mean Absolute Error: {mae}')

    # MAPE
    mape = MAPE(test_labels, y_pred.reshape(-1,1))
    print(f'Mean Absolute Percentage Error: {mape}')
    print()

    fold_perf.append((r2,mse,mae,mape))

  return fold_perf

################################ Logistic Regression Functions ################################

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def train_logistic(X, y, batch_size = 32, lr = 0.001, iterations = 1000, pr=False):
  # dimensions of the dataset
  r, c = X.shape

  # initialize weights using gaussian
  weights = np.random.normal(0,0.01,(c, 1))
  data = np.hstack((X,y.reshape(-1,1)))

  for i in range(1,iterations+1):
    if (pr and i%100 == 0):
      print(f'Iteration {i}:')
      print(weights)
      print()

    # randomly shuffle data
    np.random.shuffle(data)

    # divide into batches
    batches = [data[j:j+batch_size] for j in range(0, r, batch_size)]
    for batch in batches:
      x_data = batch[:, :-1]
      y_data = batch[:, -1].reshape(-1,1)
      weights += lr * np.dot(x_data.T, (y_data -  sigmoid(np.dot(x_data, weights))))

  return weights

def metrics_logisitic(actual, predicted):
  TN = sum(predicted[np.where(predicted == actual)]==0)
  FP = sum(predicted[np.where(predicted != actual)]==1)
  FN = sum(predicted[np.where(predicted != actual)]==0)
  TP = sum(predicted[np.where(predicted == actual)]==1)

  accuracy = np.around((TN+TP)/(TN+TP+FN+FP)*100, decimals=2)
  precision = TP/(TP+FP)
  recall = TP/(TP+FN)
  NPV = TN/(TN+FN)
  FDR = FP / (FP+TP)
  FPR = FP / (FP+TN)
  F1 = 2*(precision*recall) / (precision+recall)
  F2 = 5*(precision*recall) / (4*precision+recall)

  # print classification matrix
  print(f'TP = {TP} FP = {FP}')
  print(f'TN = {TN} FN = {FN}')
  
  # print metrics
  print(f'Accuracy = {accuracy}%')
  print(f'Precision = {precision}')
  print(f'Recall = {recall}')
  print(f'NPV = {NPV}')
  print(f'FDR = {FDR}')
  print(f'FPR = {FPR}')
  print(f'F1 score = {F1}')
  print(f'F2 score = {F2}')
  
def predict_logistic(test_data, test_labels, weights):
  predictions = np.around(sigmoid(np.dot(test_data,weights)))
  metrics_logisitic(test_labels.reshape(-1,1), predictions)

################################ Support Vector Machines Functions ################################

# Stratified K fold cross validation
def stratified_k_fold(X,y, k=5):
  # stratify the dataset
  full_data = np.hstack((X,y))
  unique_labels = np.unique(y)

  #  accumulate labels in dataset
  labels_data = []
  for ul in unique_labels:
    labels_data.append(np.random.permutation(full_data[np.where(full_data[:,-1] == ul)]))

  # divide labels for each fold as evenly as possible
  stratified_data = []
  for i in range(k):
    loc_data = []
    for ld in labels_data:
      slice = ld.shape[0]//k
      loc_data.append(ld[i*slice:slice*(i+1),:])
    stratified_data.append(np.concatenate(loc_data[:]))

  stratified_data = np.asarray(stratified_data)

  return stratified_data

# split into train, validate, test
def train_val_test(stratified_data, iter, k=5):
  val_index = (iter+1) % 5
  train = [stratified_data[x] for x in range(len(stratified_data)) if x != iter and x != val_index]
  train = np.concatenate(train[:])
  test = stratified_data[iter]
  validate = stratified_data[val_index]
  return train, validate, test

def svm_train(train, test, C=1, gamma='scale', kernel='linear'):
  model = svm.SVC(C=C, gamma=gamma, kernel=kernel)
  model.fit(train[:,:-1], train[:,-1])
  prediction = model.predict(test[:,:-1])
  return np.mean(prediction==test[:,-1])*100

def grid_search_linear(stratified_data, C_list):
  accuracies = []
  for c in C_list:
    loc_accuracies = []
    for i in range(len(stratified_data)):
      train, val, test = train_val_test(stratified_data, i)
      loc_accuracies.append(svm_train(train, val, C=c))
    accuracies.append(np.mean(loc_accuracies))
  best_c = C_list[np.argmax(accuracies)]
  return best_c

def grid_search_non_linear(stratified_data, C_list, G_list):
  accuracies = []
  for g in G_list:
    for c in C_list:
      loc_accuracies = []
      for i in range(len(stratified_data)):
        train, val, test = train_val_test(stratified_data, i)
        loc_accuracies.append(svm_train(train, val, C=c, gamma=g, kernel='rbf'))
    accuracies.append(np.mean(loc_accuracies))
  best_c = C_list[np.argmax(accuracies)]
  best_g = G_list[np.argmax(accuracies)]
  return best_c, best_g

def main():
  ################################ Principal Component Analysis ################################
  data_root = './'
  q1_path = os.path.join(data_root, 'van_gogh')

  X_orig = []
  for img in os.listdir(q1_path):
    X_orig.append(mpimg.imread(q1_path + "/" + img, format="JPG"))
  # stack grayscale images along third dimension and flatten to (4096,3)
  X_proc = X_orig.copy()
  for i in range(len(X_proc)):
    if (X_proc[i].shape == (64,64)):
      X_proc[i] = np.dstack((X_proc[i],X_proc[i],X_proc[i]))
    X_proc[i] = X_proc[i].reshape(-1, X_proc[i].shape[-1])

  # create 3D array by stacking flattened matrices
  X_proc = np.dstack([arr for arr in X_proc]).reshape((877,4096,3))

  # slicing to obtain each color channel matrix separately
  X1 = X_proc[:,:,0]
  X2 = X_proc[:,:,1]
  X3 = X_proc[:,:,2]

  print('########### PCA Using SVD without noise ###########')
  print(f'X1 PVE: {PVE(SVD(X1))[:10]}')
  print(f'X2 PVE: {PVE(SVD(X2))[:10]}')
  print(f'X3 PVE: {PVE(SVD(X3))[:10]}')

  plt.figure(0, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X1)))
  plt.title('PVE vs PCA components for 1st Channel')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()

  print()

  plt.figure(1, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X2)))
  plt.title('PVE vs PCA components for 2nd Channel')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()
    
  print()

  plt.figure(2, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X3)))
  plt.title('PVE vs PCA components for 3rd Channel')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()

  # calculate the mean and variance of the dataset
  X_mean = np.mean(X_proc)
  X_var = np.var(X_proc)

  # sampling from a 64x64x3 gaussian
  gaussian_noise = np.random.normal(X_mean,X_var,(64,64,3))

  # scale gaussian noise by 0.01
  gaussian_noise = gaussian_noise + (gaussian_noise * 0.01)
  # add noise to the dataset
  X_noise = X_orig.copy()

  for i in range(len(X_noise)):
    if (X_noise[i].shape == (64,64)):
      X_noise[i] = np.dstack((X_noise[i],X_noise[i],X_noise[i]))
    # add noise
    X_noise[i] = X_noise[i] + gaussian_noise
    X_noise[i] = X_noise[i].reshape(-1, X_noise[i].shape[-1])

  # create 3D array by stacking flattened matrices
  X_noise = np.dstack([arr for arr in X_noise]).reshape((877,4096,3))

  # slicing to obtain each color channel matrix separately
  X1_noise = X_noise[:,:,0]
  X2_noise = X_noise[:,:,1]
  X3_noise = X_noise[:,:,2]

  print('########### PCA Using SVD with noise ###########')
  print(f'X1 noisy PVE: {PVE(SVD(X1_noise))[:10]}')
  print(f'X2 noisy PVE: {PVE(SVD(X2_noise))[:10]}')
  print(f'X3 noisy PVE: {PVE(SVD(X3_noise))[:10]}')

  plt.figure(3, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X1_noise)))
  plt.title('PVE vs PCA components for 1st Channel with noise')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()

  print()

  plt.figure(4, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X2_noise)))
  plt.title('PVE vs PCA components for 2nd Channel with noise')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()

  print()

  plt.figure(5, figsize=(10,5))
  plt.bar(x=[i for i in range(1,101)], height= PVE(SVD(X3_noise)))
  plt.title('PVE vs PCA components for 3rd Channel with noise')
  plt.xlabel('Components')
  plt.ylabel('PVE')
  plt.tight_layout()
  plt.show()

  ################################ Linear Regression ################################
  # read in the dataset
  admissions_data = np.genfromtxt('./q2_dataset.csv',delimiter=',', skip_header =1)

  # normalize the features
  norm = np.linalg.norm(admissions_data[:,:-1])
  admissions_data[:,:-1] = admissions_data[:,:-1]/norm

  # linear train and test
  train, test = k_fold_validate(admissions_data)
  weights = train_linear(train)
  print('########### Linear Regression Fold Metrics ###########')
  fold_perf_linear = predict(weights, test)

  # lasso train and test
  weights, test = lasso_gradient_descent(admissions_data)
  print('########### Linear Regression with Lasso Fold Metrics ###########')
  fold_perf_lasso = predict(weights, test)

  # boxplots for each metric to compare the two models
  plt.figure(6)
  plt.boxplot([[k[0] for k in fold_perf_linear],[k[0] for k in fold_perf_lasso]],showmeans=True, labels=['Linear R^2', 'Lasso R^2'])
  plt.show()
  plt.figure(7)
  plt.boxplot([[k[1] for k in fold_perf_linear],[k[1] for k in fold_perf_lasso]],showmeans=True, labels=['Linear MSE','Lasso MSE'])
  plt.show()
  plt.figure(8)
  plt.boxplot([[k[2] for k in fold_perf_linear],[k[2] for k in fold_perf_lasso]],showmeans=True, labels=['Linear MAE', 'Lasso MAE'])
  plt.show()
  plt.figure(9)
  plt.boxplot([[k[3] for k in fold_perf_linear],[k[3] for k in fold_perf_lasso]],showmeans=True, labels=['Linear MAPE','Lasso MAPE'])
  plt.show()

  ################################ Logistic Regression ################################
  # read in train and test sets
  titanic_train_set = pd.read_csv('./q3_train_dataset.csv')
  titanic_test_set = pd.read_csv('./q3_test_dataset.csv')

  # convert string values to int
  titanic_train_set['Gender'] = titanic_train_set['Gender'].map({'male':0, 'female':1}).astype(int)
  titanic_train_set['Port of Embarkation'] = titanic_train_set['Port of Embarkation'].map({'S':0, 'C':1,'Q':2}).astype(int)
  titanic_test_set['Gender'] = titanic_test_set['Gender'].map({'male':0, 'female':1}).astype(int)
  titanic_test_set['Port of Embarkation'] = titanic_test_set['Port of Embarkation'].map({'S':0, 'C':1,'Q':2}).astype(int)

  # extract numpy array from pandas
  titanic_train_set = titanic_train_set.values
  titanic_test_set = titanic_test_set.values

  # normalize train features
  norm = np.linalg.norm(titanic_train_set[:,1:])
  titanic_train_set[:,1:] = titanic_train_set[:,1:]/norm

  # normalzie test features
  norm = np.linalg.norm(titanic_test_set[:,1:])
  titanic_test_set[:,1:] = titanic_test_set[:,1:]/norm

  # get get the features and labels for the train and test sets
  titanic_train_features = titanic_train_set[:,1:]
  titanic_test_features = titanic_test_set[:,1:]
  titanic_train_labels = titanic_train_set[:,0]
  titanic_test_labels = titanic_test_set[:,0]

  # Mini-batch with batch_size 32
  start = time.time()
  weights_mini = train_logistic(titanic_train_features, titanic_train_labels)
  end = time.time()
  print('########### Mini-batch Logistic Regression Metrics ###########')
  print(f'Mini-batch train time: {end - start}s')
  predict_logistic(titanic_test_features, titanic_test_labels, weights_mini)

  print()

  # Stochastic
  start = time.time()
  weights_stochastic = train_logistic(titanic_train_features, titanic_train_labels, batch_size=1)
  end = time.time()
  print('########### Stochastic Logistic Regression Metrics ###########')
  print(f'Stochastic train time: {end - start}s')
  predict_logistic(titanic_test_features, titanic_test_labels, weights_stochastic)

  print()

  # full batch
  start = time.time()
  weights_full = train_logistic(titanic_train_features, titanic_train_labels, batch_size=titanic_train_features.shape[0], pr=True)
  end = time.time()
  print('########### Full-batch Logistic Regression Weights and Metrics ###########')
  print(f'Full-batch train time: {end - start}s')
  predict_logistic(titanic_test_features, titanic_test_labels, weights_full)

  ################################ Support Vector Machines ################################
  flowers_data = loadmat('./q4_dataset.mat')
  flowers_images = flowers_data['images']
  flowers_inception = flowers_data['inception_features']
  flowers_y = flowers_data['class_labels']

  # the parameter options
  C_list = [10**-6, 10**-4, 10**-2, 1, 10, 10**10]
  G_list = [2**-4,2**-2,1,2**2,2**10,'scale']

  # get stratified data 
  stratified_data = stratified_k_fold(flowers_inception,flowers_y)

  # get best C parameter
  start = time.time()
  best_c = grid_search_linear(stratified_data, C_list)
  end = time.time()
  print('########### Grid Search for Parameter C ###########')
  print(f'Grid-Search for C time: {end-start}s')
  print(f'Best C: {best_c}')

  # test with best C
  loc_accuracies_linear = []
  for i in range(len(stratified_data)):
    train, val, test = train_val_test(stratified_data, i)
    train = np.concatenate((train,val))
    loc_accuracies_linear.append(svm_train(train, test, C=best_c))
  accuracy_linear = (np.mean(loc_accuracies_linear))
  print('########### Mean Linear Kernel Accuracy ###########')
  print(f'Mean Accuracy Linear {accuracy_linear}%')

  # get best C and gamma parameter combination
  start = time.time()
  best_c, best_g = grid_search_non_linear(stratified_data, C_list, G_list)
  end = time.time()
  print('########### Grid search for C and Gamma parameters ###########')
  print(f'Grid-Search for C  and gamma time: {end-start}s')
  print(f'Best C: {best_c}, Best G: {best_g}')

  # test with best C and best gamma
  loc_accuracies_rbf = []
  for i in range(len(stratified_data)):
    train, val, test = train_val_test(stratified_data, i)
    train = np.concatenate((train,val))
    loc_accuracies_rbf.append(svm_train(train, test, C=best_c, gamma=best_g, kernel='rbf'))
  accuracy_rbf = (np.mean(loc_accuracies_rbf))
  print('########### Mean RBF Kernel Accuracy ###########')
  print(f'Mean Accuracy RBF {accuracy_rbf}%')

  # plot accuracy boxplots for both kernels
  plt.figure(10)
  plt.boxplot([loc_accuracies_linear,loc_accuracies_rbf], showmeans=True, labels=['Linear Accuracy', 'RBF Accuracy'])
  plt.show()

if __name__ == "__main__":
    main()
