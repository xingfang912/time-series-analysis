# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:54:36 2017

@author: xing-deeplearning

Bidirectional Recurrent Deep Neural Network with LSTM for time series
"""

import numpy as np
#import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Data = []; counter = 0
file_name = 'IV'
with open(r"/home/xing/Documents/time series/"+file_name) as f:
    for line in f:
        counter += 1
        line = line.strip()
        if line != '':
            s = line.split()
            Data.append(float(s[1]))

# Data pre-processing
n_steps = 20 # the size of each mini-batch
n_inputs = 1
n_neurons = 100
n_outputs = 1 #This controls the predicting steps
n_layers = 3
MAX = max(Data)
print "Max: %f"%MAX
# Normalization
Data = [i/MAX for i in Data]

testing_points = 100
splitting_point = len(Data) - (testing_points+n_outputs)


training_data = Data[:splitting_point-n_outputs]
testing_data = Data[splitting_point:-n_outputs]
labels = Data[1:splitting_point]
training_labels = []; testing_labels = []
for i in range(len(labels)-(n_outputs-1)):
    temp = []
    for j in range(n_outputs):
        temp.append(labels[i+j])
    training_labels.append(temp)


labels = Data[splitting_point+1:]
for i in range(len(labels)-(n_outputs-1)):
    temp = []
    for j in range(n_outputs):
        temp.append(labels[i+j])
    testing_labels.append(temp)

# # plotting stuff
# #plt.plot(np.arange(splitting_point),training_labels)
# plt.plot(np.arange(testing_size-1),np.array(testing_labels)*MAX)

training_data = np.array([[[training_data[j]] for j in range(i*n_steps,(i+1)*n_steps)] for i in range(int((splitting_point-n_outputs)/n_steps))])
training_labels = np.array([[training_labels[j] for j in range(i*n_steps,(i+1)*n_steps)] for i in range(int((splitting_point-n_outputs)/n_steps))])
testing_data = np.array(testing_data).reshape(-1,n_steps,n_inputs)
testing_labels = np.array(testing_labels).reshape(-1,n_steps,n_outputs)

#Time series prediction
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
X_reversed = tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_steps,n_outputs])
#cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
#                                              output_size = n_outputs)

#basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)


'''
Dropout
'''
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons),
#                                                                              input_keep_prob=0.5) for _ in range(n_layers)])


'''
LSTM
'''
multi_layer_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for _ in range(n_layers)])
multi_layer_cell_reversed = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons) for _ in range(n_layers)])

rnn_outputs,states = tf.nn.bidirectional_dynamic_rnn(multi_layer_cell,multi_layer_cell_reversed,X,dtype=tf.float32)
rnn_outputs_fw,rnn_outputs_bw = rnn_outputs
stacked_rnn_outputs_fw = tf.reshape(rnn_outputs_fw,[-1,n_neurons])
stacked_rnn_outputs_bw = tf.reshape(rnn_outputs_bw,[-1,n_neurons])
stacked_rnn_outputs = tf.add(stacked_rnn_outputs_fw,stacked_rnn_outputs_bw)
stacked_outputs = fully_connected(stacked_rnn_outputs,n_outputs,activation_fn=None)
# stacked_outputs = fully_connected(stacked_rnn_outputs,n_outputs) #This will force the output to be positive numbers
outputs = tf.reshape(stacked_outputs,[-1,n_steps,n_outputs])

learning_rate = 0.001
#loss = tf.nn.l2_loss(tf.reduce_mean(tf.square(outputs-y))+ sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
loss = tf.add_n([tf.reduce_mean(tf.square(outputs-y))]+ tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_iterations = 10000
# batch_size = 15
saver = tf.train.Saver(max_to_keep=1)
best_mse = 9999
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch = training_data
        y_batch = training_labels
        sess.run(training_op, feed_dict={X:X_batch,y:y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print (iteration, "\tMSE:", mse)
            if mse < best_mse:
                best_mse = mse
                saver.save(sess,r'/home/xing/Documents/time series/models/model-bi-'+file_name,global_step=iteration,write_meta_graph=True)
                print ("Model saved.")
                it = iteration

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(r'/home/xing/Documents/time series/models/model-bi-'+file_name+'-'+str(it)+'.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/xing/Documents/time series/models/'))
    y_pred = sess.run(outputs,feed_dict={X:testing_data})
    mse = sess.run(loss,feed_dict={X:testing_data, y:testing_labels})

#print ("Training MSE: "+str(best_mse))
print ("Testing MSE: "+str(mse))

if n_outputs == 1:
    y_pred_list = y_pred.reshape(testing_points*n_outputs)
    # plt.plot(np.arange(len(y_pred)),y_pred*MAX)


    ground_truth_list = list(testing_labels.reshape(testing_points*n_outputs))
    D = 0
    N = 0
    for i in range(len(y_pred_list)):
        if y_pred_list[i] < 0: #used for V predicting 10 steps
            continue
        else:
            N += abs(y_pred_list[i]*MAX-ground_truth_list[i]*MAX)
            D += ground_truth_list[i]*MAX
    print ("Testing PMAD: "+str(N/D))

else:
    print ("Testing PMAD: ")
    y_pred_list = y_pred.reshape(-1,n_outputs)
    ground_truth = testing_labels.reshape(-1,n_outputs)
    for j in range(n_outputs):
        D = 0; N = 0
        for i in range(len(y_pred_list[:,j])):
            if y_pred_list[i][j] < 0:
                continue
            else:
                N += abs(y_pred_list[i][j]*MAX-ground_truth[i][j]*MAX)
                D += ground_truth[i][j]*MAX
        print N/D,

print ""
print "Writing to files..."
with open(file_name+"_ground_truth_"+str(n_outputs)+"_steps_"+str(n_steps)+"_batch-size_"+str(n_layers)+"_layers.txt","w") as f:
    for i in range(testing_labels.shape[0]):
        for j in range(testing_labels.shape[1]):
            for k in range(testing_labels.shape[2]):
                f.write(str(testing_labels[i][j][k]*MAX)+" ")
            f.write("\n")

with open(file_name+"_predictions_"+str(n_outputs)+"_steps_"+str(n_steps)+"_batch-size_"+str(n_layers)+"_layers.txt","w") as f:
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            for k in range(y_pred.shape[2]):
                f.write(str(y_pred[i][j][k]*MAX)+" ")
            f.write("\n")

print "Complete!"
