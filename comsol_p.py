import mnist_loader
import numpy as np
import math
from dnn import Network

def write_file(image, name, layer):
    def h(phi):
        lmda = 7.5e-4
        n = 1.7227
        h = lmda*phi/n
        return h+5e-4

    delta = 4e-4

    f=open(name,"a+")

    for i in range(784):
        x = i%28
        y = 27 -p//28
        f.write(str(delta) + " " + str(delta) + " " +str(h(image[i])) + " " + str(delta*x) + " " + str(delta*y) + " " + str(layer*0.01) + "\n")

    f.close()

training_data1, training_data2, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data1 = list(training_data1)
training_data2 = list(training_data2)
test_data = list(test_data)
net = Network(size=1)
net.SGD(training_data1[:1000], training_data2[:1000], test_data[:1], epochs=30, mini_batch_size=5, eta=0.0001, beta1=0.9, beta2=0.999)

layer = 1
for theta in net.theta:
    name = "data.txt"
    write_file(theta, name, layer)
    layer += 1

tr_d, va_d, te_d = mnist_loader.load_data()
x = tr_d[0]
y = tr_d[1]

i = 0
right_num = 0
while right_num < 1:
    if np.argmax(net.feedforward(x[i])) == y[i]:
        write_file(x[i], "digit{}.txt".format(right_num), 0)
        right_num += 1
    i += 1
