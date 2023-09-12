from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
from train import model
from preprocess import X, y, X_train, X_test, y_train, y_test

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


ytrue_full = np.argmax(y_train, axis=1).tolist()


#all numbers should be in top left or bottom right 
print(multilabel_confusion_matrix(ytrue,yhat))

print(accuracy_score(ytrue, yhat))