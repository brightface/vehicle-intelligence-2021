import numpy as np
import random
from math import sqrt, pi, exp

def gaussian_prob(obs, mu, sig):
    # Calculate Gaussian probability given
    # - observation
    # - mean
    # - standard deviation
    num = (obs - mu) ** 2
    denum = 2 * sig ** 2
    norm = 1 / sqrt(2 * pi * sig ** 2)
    return norm * exp(-num / denum)

# Gaussian Naive Bayes class
class GNB():
    # Initialize classification categories
    def __init__(self):
        self.classes = ['left', 'keep', 'right']

    # Given a set of variables, preprocess them for feature engineering.
    def process_vars(self, vars):
        # The following implementation simply extracts the four raw values
        # given by the input data, i.e. s, d, s_dot, and d_dot.
        s, d, s_dot, d_dot = vars
        return s, d, s_dot, d_dot

    # Train the GNB using a combination of X and Y, where
    # X denotes the observations (here we have four variables for each) and
    # Y denotes the corresponding labels ("left", "keep", "right").
    def train(self, X, Y):
        '''
        Collect the data and calculate mean and standard variation
        for each class. Record them for later use in prediction.
        '''
        # TODO: implement code.
        ### Collect the data ###
        values_by_label = dict()

        # INPUT FORMAT [ s, d, s_dot, d_dot ]
        # for x,y in zip(X,Y):
        #     print(x)
        #     print(y)

        for c in self.classes:
            values_by_label[c] = np.empty((4, 0))
        # Collect the data
        # Dict ==> { class : [s : ], [d%4 : ], [s_dot : ], [d_dot : ]}
        for x, y in zip(X, Y):
            # item [s, d%4, s_dot, d_dot]
            item = np.array([[x[0]], [(x[1] % 4)], [x[2]], [x[3]]])
            # concat the array for get mean, stddevs
            values_by_label[y] = np.append(values_by_label[y], item, axis=1)

        # print(values_by_label)

        ### Caculate mean and stddevs for each classes ###
        means = dict()
        stddevs = dict()

        for c in self.classes:
            class_array = np.array(values_by_label[c])
            # get mean
            means[c] = np.mean(class_array, axis=1)
            # get stddevs
            stddevs[c] = np.std(class_array, axis=1)

        self.means = means
        self.stddevs = stddevs
    # Given an observation (s, s_dot, d, d_dot), predict which behaviour
    # the vehicle is going to take using GNB.
    def predict(self, observation):
        '''
        Calculate Gaussian probability for each variable based on the
        mean and standard deviation calculated in the training process.
        Multiply all the probabilities for variables, and then
        normalize them to get conditional probabilities.
        Return the label for the highest conditional probability.
        '''
        # TODO: implement code.
        # Calculate Gaussian probability for each variable
        probs = dict()

        for c in self.classes:
            cur_prob = 1.00
            # Multiply all the probabilities for variables
            for idx in range(len(observation)):
                cur_prob *= gaussian_prob(observation[idx], self.means[c][idx], self.stddevs[c][idx])

            probs[c] = cur_prob

        # Get hightest conditinal probability & label
        highest_prob = 0
        highest_class = "keep"

        for c in self.classes:
            if probs[c] > highest_prob:
                highest_prob = probs[c]
                highest_class = c

        return highest_class


