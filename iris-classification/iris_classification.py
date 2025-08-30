import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix 

# Variant 1:
# A neural network for classification (MultiLayerPerceptron) is formed with the following properties:
# an input layer with 4 neurons representing the characteristics of the iris plant;
# a hidden layer with 10 neurons
# an output layer with 3 neurons representing the classes to be recognized
# Variant 2:
# In the second variant, two hidden layers with five and three neurons respectively are used
# The activation function tanh and the optimizer adam are used
MODEL_VARIANT = 2

# load iris-dataset
data_train = pd.read_csv('./iris.csv')

# map iris species to numbers
data_train.loc[data_train['species'] == 'Iris-setosa', 'species'] = 0
data_train.loc[data_train['species'] == 'Iris-versicolor', 'species'] = 1
data_train.loc[data_train['species'] == 'Iris-virginica', 'species'] = 2
data_train = data_train.apply(pd.to_numeric)

# convert the loaded data into a matrix representation
data_train_array =  data_train.to_numpy()

# add a set seed to ensure the reproducibility of the results
np.random.seed(17)

# Split loaded data into 2 catagories: testdata and trainsingsdata

# Use 80% to train the model and 20% for testing the model
# X_ for input
# y_for output
X_train, X_test, y_train, y_test = train_test_split(data_train_array[:,:4],
                                                    data_train_array[:,4],
                                                    test_size=0.2)

if MODEL_VARIANT == 1:
    mlp = MLPClassifier(hidden_layer_sizes=(10,),activation='relu', solver='adam', max_iter=350, batch_size=10, verbose=True)
elif MODEL_VARIANT == 2:
    mlp = MLPClassifier(hidden_layer_sizes=(5,3),activation='tanh', solver='adam', max_iter=350, batch_size=10, verbose=True)

# The neural network is trained with the training data
mlp.fit(X_train, y_train)

print("Trainingresult: %5.3f" % mlp.score(X_train, y_train))

# The model is evaluated using the test data.
predictions = mlp.predict(X_test)
# and the confusion matrix is output
print(confusion_matrix(y_test,predictions)) 

# Precision, recall, and f1-score are calculated from the confusion matrix and output
print(classification_report(y_test,predictions)) 

# The model is tested and the result is output
print("Testergebnis: %5.3f" % mlp.score(X_test,y_test))

# The following outputs the values of the weights per layer
print("WEIGHTS:", mlp.coefs_)
print("BIASES:", mlp.intercepts_) 

# The model is applied, for example, to predict the following values from the test set with the characteristics [sepal-length, sepal-width, 
# petal-length, petal-width]
print(mlp.predict([[5.1,3.5,1.4,0.2], [5.9,3.,5.1,1.8], [4.9,3.,1.4,0.2], [5.8,2.7,4.1,1.]]))

# The loss curve is visualized and saved in the file Plot_of_loss_values.png in PNG format
loss_values = mlp.loss_curve_
plt.plot(loss_values)
plt.savefig(f"./Plot_of_loss_values_{MODEL_VARIANT}.png")
plt.show()