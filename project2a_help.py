from load import mnist
import numpy as np
import pylab

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 2 convolution layer, 2  max pooling layer and a softmax layer


np.random.seed(10)
batch_size = 128
noIters = 50

def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
    pool_dim = (2, 2)

    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    o1 = pool.pool_2d(y1, pool_dim, ignore_border=True)

    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    o2 = pool.pool_2d(y2, pool_dim, ignore_border=True)

    o2f = T.flatten(o2, outdim=2)

    y3 = T.nnet.relu(T.dot(o2f,w3) + b3)

    y4 = T.nnet.softmax(T.dot(y3, w4) + b4)
    return y1, o1, y2, o2, y3, y4

def sgd(cost, params, lr=0.05, decay=0.0001):
    grads = T.grad(cost=cost, wrt=params)

    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.5):
    grads = T.grad(cost=cost, wrt=params)

    updates = []
    for p,g in zip(params, grads):
        v=theano.shared(p.get_value()*0)
        v_new = momentum*v-(g + decay*p) * lr
        updates.append([p,p+v_new])
        updates.append([v,v_new])
    return updates

def sgd_rmsprop(cost, params, lr=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)

    updates = []
    for p,g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1-rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr *(g+decay*p)))
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)
#TODO change this to run nn faster
trainNumber = 60000
testNumber  = 10000

trX, trY = trX[:trainNumber], trY[:trainNumber]
teX, teY = teX[:testNumber], teY[:testNumber]


X = T.tensor4('X')
Y = T.matrix('Y')

num_filters_1 = 15
num_filters_2 = 20

w1, b1 = init_weights_bias4((num_filters_1, 1, 9, 9), X.dtype)
w2, b2 = init_weights_bias4((num_filters_2, num_filters_1, 5, 5), X.dtype)
w3, b3 = init_weights_bias2((num_filters_2*3*3, 100), X.dtype)
w4, b4 = init_weights_bias2((100,                10), X.dtype)
y1, o1, y2, o2, y3, y4  = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(y4, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(y4, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]

#updates = sgd(cost, params, lr=0.05)
updates = sgd_momentum(cost, params, lr=0.05)
#updates = sgd_rmsprop(cost, params, lr=0.05)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1, y2, o2], allow_input_downcast=True)

test_accuracy_array = []
training_cost_array =[]
total = 0
for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
        total = total+cost
    training_cost_array.append(total)
    test_accuracy_array.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    print(test_accuracy_array[i])

pylab.figure()
pylab.plot(range(noIters), test_accuracy_array)
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2a_1.png')

pylab.figure()
pylab.plot(range(noIters), training_cost_array)
pylab.xlabel('epochs')
pylab.ylabel('training cost')
pylab.savefig('figure_2a_2.png')




w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(num_filters_1):
    pylab.subplot(5, 5, i+1); 
    pylab.axis('off'); 
    pylab.imshow(w[i,:,:,:].reshape(9,9))
#pylab.title('filters learned')
pylab.savefig('figure_2a_3.png')

ind = np.random.randint(low=0, high=testNumber)
convolved1, pooled1, convolved2, pooled2 = test(teX[ind:ind+1,:])

pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
#pylab.title('input image')
pylab.savefig('figure_2a_4.png')

pylab.figure()
pylab.gray()
for i in range(num_filters_1):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved1[0,i,:].reshape(20,20))
#pylab.title('convolved feature maps in first convolution layer')
pylab.savefig('figure_2a_5.png')

pylab.figure()
pylab.gray()
for i in range(num_filters_1):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled1[0,i,:].reshape(10,10))
#pylab.title('pooled feature maps in first convolution layer')
pylab.savefig('figure_2a_6.png')

pylab.figure()
pylab.gray()
for i in range(num_filters_2):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved2[0,i,:].reshape(6,6))
#pylab.title('convolved feature maps in second convolution layer')
pylab.savefig('figure_2a_7.png')

pylab.figure()
pylab.gray()
for i in range(num_filters_2):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled2[0,i,:].reshape(3,3))
#pylab.title('pooled feature maps in second convolution layer')
pylab.savefig('figure_2a_8.png')

#pylab.show()
