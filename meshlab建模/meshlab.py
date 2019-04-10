import mnist_loader
import numpy as np
import math
from dnn import Network

def write_file(image, name):
    def h(phi):
        lmda = 7.5e-4
        n = 1.7227
        h = lmda*phi/n
        return h+5e-4

    delta = 4e-4
    f=open(name,"a+")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for i in range(1,20):
                for j in range(1,20):
                    f.write(str((x+i/20)*delta) + " " + str((y+j/20)*delta) + " " +str(h(image[x,y])) + " " + str(0) + " " + str(0) + " " + str(1) + "\n")
                    f.write(str((x+i/20)*delta) + " " + str((y+j/20)*delta) + " " +str(0.0)+ " " + str(0) + " " + str(0) + " " + str(-1) + "\n")

                    if x == 0 and i == 1:
                        for n in range(1,20):
                            f.write(str(x) + " " + str((y+j/20)*delta) + " " +str(h(image[x,y])/20*n)+ " " + str(-1) + " " + str(0) + " " + str(0) + "\n")
                    if x == 27 and i == 19:
                        for n in range(1,20):
                            f.write(str((x+1)*delta) + " " + str((y+j/20)*delta) + " " + str(h(image[x,y])/20*n)+ " " + str(1) + " " + str(0) + " " + str(0) + "\n")
                    if y == 0 and j == 1:
                        for n in range(1,20):
                            f.write(str((x+i/20)*delta) + " " + str(y) + " " +str(h(image[x,y])/20*n)+ " " + str(0) + " " + str(-1) + " " + str(0) + "\n")
                    if y == 27 and j == 19:
                        for n in range(1,20):
                            f.write(str((x+i/20)*delta) + " " + str((y+1)*delta) + " " + str(h(image[x,y])/20*n)+ " " + str(0) + " " + str(1) + " " + str(0) + "\n")

                    if not x == 27 and i == 19:
                        if not (image[x,y] == image[x+1,y]):
                            for n in range(1,10):
                                f.write(str((x+1)*delta) + " " + str((y+j/20)*delta) + " " + str(n/10*h(image[x,y])+(10-n)/10*h(image[x+1,y])) + " " + str(np.sign(image[x,y]-image[x+1,y])) + " " + str(0) + " " + str(0) + "\n")
                        else:
                            f.write(str((x+1)*delta) + " " + str((y+j/20)*delta) + " " + str(h(image[x,y])) + str(0) + " " + str(0) + " " + str(1) + "\n")
                    if not y == 27 and j == 19:
                        if not (image[x,y] == image[x,y+1]):
                            for n in range(1,10):
                                f.write(str((x+i/20)*delta) + " " + str((y+1)*delta) + " " + str(n/10*h(image[x,y])+(10-n)/10*h(image[x,y+1])) + " " + str(0) + " " + str(np.sign(image[x,y]-image[x,y+1])) + " " + str(0) + "\n")
                        else:
                            f.write(str((x+i/20)*delta) + " " + str((y+1)*delta) + " " + str(h(image[x,y])) + str(0) + " " + str(0) + " " + str(1) + "\n")

    f.close()
"""
tr_d, va_d, te_d = mnist_loader.load_data()
image = tr_d[0]
result = tr_d[1]

image0 = np.reshape(image[0], (28,28))
write_file(image0, "digit.txt")
"""

training_data1, training_data2, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data1 = list(training_data1)
training_data2 = list(training_data2)
test_data = list(test_data)
net = Network(size=2)
net.SGD(training_data1, training_data2[:100], test_data[:100],
        epochs=1, mini_batch_size=10, eta=0.001, beta1=0.9, beta2=0.999)

num = 1
for theta in net.theta:
    name = "layer_{}.txt".format(num)
    theta = np.reshape(theta, (28,28))
    write_file(theta, name)
    num += 1
