# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:00:22 2020

@author: sssha
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

def get_dataset(training=True): 
  mnist = keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  if training == True:
    return (np.asarray(train_images), np.asarray(train_labels))
  else:
    return (np.asarray(test_images), np.asarray(test_labels))

def print_stats(train_images, train_labels):
  print(len(train_images))
  tup = train_images.shape
  print(str(tup[1]) + str('x') + str(tup[2]))
  class_names = {0 : [0,'Zero'], 1 : [0,'One'], 2: [0,'Two'], 3 : [0,'Three'], 4 : [0,'Four'], 5 : [0,'Five'],  6: [0, 'Six'], 7 : [0, 'Seven'], 8 : [0, 'Eight'], 9 : [0, 'Nine']}
  for i in range(len(train_labels)):
    for num in class_names.keys():
      if train_labels[i] == num:
        class_names[num][0] += 1
  j = 0
  for key in class_names.keys():
    print(str(j) + str("."), class_names[key][1], str("-"), class_names[key][0])
    j += 1
    
def build_model():
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dense(64, activation='relu'))
  model.add(keras.layers.Dense(10))
  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=keras.optimizers.SGD(learning_rate=0.001),metrics=['accuracy'])
  return model

def train_model(model, train_images, train_labels, T):
  model.fit(x=train_images, y=train_labels, epochs = T)
    
def evaluate_model(model, test_images, test_labels, show_loss=True):
  test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
  if show_loss==True:
    print("Loss:", "{:.4f}".format(test_loss)) 
    print(str("Accuracy: ") + str("{:.2f}".format(test_accuracy * 100))+ str('%'))
  else:
    print(str("Accuracy: ")+ str("{:.2f}".format(test_accuracy * 100)) + str('%'))
    
def predict_label(model, test_images, index):
   prediction = model.predict(test_images)
   label_pred = prediction[index]
   class_labels = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
   li = sorted(zip(label_pred, class_labels), reverse=True)[:3]
   print(str(li[0][1])+ str(": ") + str("{:.2f}".format(li[0][0] * 100)) + str('%'))
   print(str(li[1][1])+ str(": ") + str("{:.2f}".format(li[1][0] * 100)) + str('%'))
   print(str(li[2][1])+ str(": ") + str("{:.2f}".format(li[2][0] * 100)) + str('%'))
  
def main():  
  layer = keras.layers.Softmax()
  (train_images, train_labels) = get_dataset()
  '''print(type(train_images))
  print(type(train_labels))
  print(type(train_labels[0]))'''
  print_stats(train_images, train_labels)
  
  model = build_model()
  train_model(model, train_images, train_labels, 10)
  (test_images, test_labels) = get_dataset(False)
  model.add(layer)
  predict_label(model, test_images, 1)

  evaluate_model(model, test_images, test_labels, False)
  #evaluate_model(model, test_images, test_labels, False)
  #evaluate_model(model, test_images, test_labels)
  #(test_images, test_labels) = get_dataset(False)
  
if __name__=='__main__':
    main()
