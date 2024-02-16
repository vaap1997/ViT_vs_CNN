import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

# Upload DataSet

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

ndata_train = x_train.shape[0]
ndata_test = x_test.shape[0]

plt.pcolor(1-x_train[9,::-1,:], cmap='gray')
plt.axis('off')

# Save sample

pd.DataFrame(x_train[9],index=None, columns=None).to_csv('example.csv', index=False, header=False)
pd.DataFrame(x_train[9]*255,index=None, columns=None).to_csv('example_255.csv', index=False, header=False)

# Reshaping it to 28x28x1

x_train = x_train.reshape((ndata_train,28,28,1))
x_test = x_test.reshape((ndata_test,28,28,1))

xshape = x_train.shape[1:4]
xshape

# Convolution Neural Net

CNNmodel = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation=tf.nn.relu,input_shape=xshape), #kernel_size : size of each filter
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (4,4),strides=2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (3,3),strides=3),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation=tf.nn.relu,kernel_regularizer = tf.keras.regularizers.l1(0.0004)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128,activation=tf.nn.relu,kernel_regularizer = tf.keras.regularizers.l1(0.0004)),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        ])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

CNNmodel.compile(optimizer=custom_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

CNNmodel.summary()

# Model Fitting
CNNmodel.fit(x_train,y_train,epochs=50,validation_split=0.2,batch_size=300) #

# Train in whole data train
CNNmodel.fit(x_train,y_train,epochs=20,validation_split=0,batch_size=300) #

# Predict x_test
pred_probs2 = CNNmodel.predict(x_test)
pred2 = np.argmax(pred_probs2, axis=1)
print("accuracy on test:", np.mean(pred2==y_test))

print(f"Number of misclassifications: {np.sum(pred2 != y_test)}")
print(classification_report(y_test, pred2, target_names=['1','2','3','4','5','6','7','8','9','10']))

# Plot confusion matrix
cm = confusion_matrix(y_test, pred2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap='viridis', values_format='d')

plt.show()

# Save model object
CNNmodel.save('CNNmodel_mnist')

CNNmodel2 = tf.keras.models.load_model('CNNmodel_mnist')
pred_probs2 = CNNmodel2.predict(x_test)
pred2 = np.argmax(pred_probs2, axis=1)
print("check accuracy on test reading saved model:", np.mean(pred2==y_test))

# Misclassified examples
well_classified_indices = np.where(pred2 == y_test)[0]
misclassified_indices = np.where(pred2 != y_test)[0]

# Plot well-classified examples
plt.figure(figsize=(20, 4))
for i, idx in enumerate(well_classified_indices[:10]):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {pred2[idx]} \n Actual: {y_test[idx]}')
    plt.axis('off')
plt.suptitle('Well-Classified Examples')
plt.show()

# Plot misclassified examples
plt.figure(figsize=(20, 4))
for i, idx in enumerate(misclassified_indices[:10]):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {pred2[idx]} \n Actual: {y_test[idx]}')
    plt.axis('off')
plt.suptitle('Misclassified Examples')
plt.show()
