import numpy as np 
def linear_reg(x_train , y_train , lr = 0.001 , num_iter= 1000):
    x_train = np.columnstack((np.ones(len(x_train)), x_train))
    w = 0.1*(np.ones(x_train.shape[1]))
    

    for iteration in range(num_iter):
        raw = np.matmul(x_train,   w)
        diff = y_train -raw
        grad = np.matmul(x_train.T, diff)
        w = lr*grad
        return w

def predict_lin_reg(w, x_test):
    
    x_test = np.column_stack((np.ones(len(x_test)), x_test))
    
    output = np.matmul(x_test, w)
    return output