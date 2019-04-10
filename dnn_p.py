import random
import math
import cmath
import numpy as np

class Network(object):
    def __init__(self, size):
        self.size = size
        self.set_weights()

        self.theta = []
        for w in self.weights:
            self.theta.append(np.random.random((w.shape[1], 1)))
        self.update_biases()

        self.first_moment_theta = [np.zeros(theta.shape) for theta in self.theta]
        self.second_moment_theta = [np.zeros(theta.shape) for theta in self.theta]

    def update_biases(self):
        self.biases = [np.exp(2j*math.pi*theta) for theta in self.theta]

    def set_weights(self):
        cache = {}
        self.weight = np.zeros((784,784), dtype='complex64')
        self.weights = []

        def w(d):
            z = 3e-2
            delta = 4e-4
            r = math.sqrt(d*(delta**2) + z**2)
            lmda = 7.5e-4
            w = z * (1/(2*math.pi*r) + 1/(1j*lmda)) * np.exp(2j*math.pi*r/lmda) / (r**2)
            return w

        for behind in range(784):
            for front in range(784):
                x_behind, y_behind = behind%28, 27-behind//28
                x_front, y_front = front%28, 27-front//28
                d = (x_behind - x_front)**2 + (y_behind - y_front)**2
                if d not in cache:
                    cache[d] = w(d)
                self.weight[behind, front] = cache[d]

        last_weights = np.array([self.weight[203],self.weight[210],self.weight[217],
                                 self.weight[397],self.weight[403],self.weight[409],self.weight[414],
                                 self.weight[595],self.weight[602],self.weight[609]])

        if self.size == 1:
            self.weights.append(last_weights)
        else:
            for i in range(self.size):
                if i == self.size-1:
                    self.weights.append(last_weights)
                else:
                    self.weights.append(self.weight)

    def feedforward(self, m):
        m = np.dot(self.weight, np.exp(2j*math.pi*m)).reshape(784,1)
        for t, w in zip(self.biases, self.weights):
            m = np.dot(w, m*t)
        s = np.real(m*np.conj(m))
        return s

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data1, training_data2, test_data,
            epochs, mini_batch_size, eta, beta1, beta2):
        n = len(training_data1)
        n_train = len(training_data2)
        n_test = len(test_data)

        t = 0
        for j in range(epochs):
            random.shuffle(training_data1)
            mini_batches = [
                training_data1[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                t += 1
                self.update_mini_batch(mini_batch, eta, beta1, beta2, t)
            print("Epoch {} : training_data {} / {}, test_data {} / {}"
                  .format(j, self.evaluate(training_data2), n_train, self.evaluate(test_data), n_test))


    def update_mini_batch(self, mini_batch, eta, beta1, beta2, t):
        images = np.zeros((784,len(mini_batch)))
        results = np.zeros((10, len(mini_batch))) 

        i = 0
        for x, y in mini_batch:
            images[:,i] = x.reshape(784)
            results[:,i] = y.reshape(10)
            i += 1

        nabla_theta = self.backprop(images, results)

        self.first_moment_theta = [beta1 * fmtheta + (1 - beta1) * ntheta/len(mini_batch)
                                   for fmtheta, ntheta in zip(self.first_moment_theta, nabla_theta)]
        self.second_moment_theta= [beta2 * smtheta + (1 - beta2) * (ntheta/len(mini_batch))**2
                                   for smtheta, ntheta in zip(self.second_moment_theta, nabla_theta)]

        first_unbias_theta = [fmtheta / (1 - beta1**t) for fmtheta in self.first_moment_theta]
        second_unbias_theta = [smtheta / (1 - beta2**t) for smtheta in self.second_moment_theta]

        self.theta = [relu(theta - eta * futheta / (np.sqrt(sutheta) + 1e-7))
                      for theta, futheta, sutheta in zip(self.theta, first_unbias_theta, second_unbias_theta)]
        self.update_biases()

    def backprop(self, x, y):
        nabla_theta = [np.zeros(theta.shape) for theta in self.theta]

        ms = [np.dot(self.weight, np.exp(2j*math.pi*x))]
        for t, w in zip(self.biases, self.weights):
            ms.append(np.dot(w, ms[-1]*t))

        for l in range(1, self.size+1):
            if l == 1:
                nabla_m = self.cost_derivative(ms[-1], y)
            else:
                nabla_m = np.dot(np.transpose(self.weights[-l+1]), nabla_m) * self.biases[-l+1]
            nabla_theta[-l] = np.sum(np.real(1j * np.dot(np.transpose(self.weights[-l]), nabla_m) * ms[-l-1] * self.biases[-l]), axis=1, keepdims=True)
        return nabla_theta

    def cost_derivative(self, m, y):
        s = np.real(m*np.conj(m))
        s_nom = s/np.sum(s, axis=0, keepdims=True)
        nabla_s_nom = s_nom - y

        nabla_s = nabla_s_nom
        for i in range(s.shape[1]):
            nabla_s[:,i] = np.dot(nom_mat(s_nom[:,i].reshape((10,1)), s[:,i].reshape((10,1))), nabla_s_nom[:,i])

        nabla_m = nabla_s * np.conj(m)
        return nabla_m

def nom_mat(s_nom, s):
    nom_mat = np.zeros((10,10))
    for i in range(10):
        nom_mat[i] = -np.transpose(s_nom)[0]
    nom_mat = (nom_mat + np.eye(10))/np.sum(s)
    return nom_mat

def relu(z):
    x = np.full(z.shape, 0.99)
    y = np.full(z.shape, 0.01)
    return np.minimum(np.maximum(z,y), x)
