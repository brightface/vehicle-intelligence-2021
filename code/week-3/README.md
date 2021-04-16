# Week 3 - Kalman Filters, EKF and Sensor Fusion

---

[//]: # (Image References)
[kalman-result]: ./kalman_filter/graph.png
[EKF-results]: ./EKF/plot.png

## Kalman Filter Example

In directory [`./kalman_filter`](./kalman_filter), a sample program for a small-scale demonstration of a Kalman filter is provided. Run the following command to test:

```
$ python testKalman.py
```

This program consists of four modules:

* `testKalman.py` is the module you want to run; it initializes a simple Kalman filter and estimates the position and velocity of an object that is assumed to move at a constant speed (but with measurement error).
* `kalman.py` implements a basic Kalman fitler as described in class.
* `plot.py` generates a plot in the format shown below.
* `data.py` provides measurement and ground truth data used in the example.

The result of running this program with test input data is illustrated below:

![Testing of Kalman Filter Example][kalman-result]

Interpretation of the above results is given in the lecture.

In addition, you can run `inputgen.py` to generate your own sample data. It will be interesting to experiment with a number of data sets with different characteristics (mainly in terms of variance, i.e., noise, involved in control and measurement).

---

## Assignment - EFK & Sensor Fusion Example

In directory [`./EKF`](./EKF), template code is provided for a simple implementation of EKF (extended Kalman filter) with sensor fusion. Run the following command to test:

```
$ python run.py
```

The program consists of five modules:

* `run.py` is the modele you want to run. It reads the input data from a text file ([data.txt](./EKF/data.txt)) and feed them to the filter; after execution summarizes the result using a 2D plot.
* `sensor_fusion.py` processees measurements by (1) adjusting the state transition matrix according to the time elapsed since the last measuremenet, and (2) setting up the process noise covariance matrix for prediction; selectively calls updated based on the measurement type (lidar or radar).
* `kalman_filter.py` implements prediction and update algorithm for EKF. All the other parts are already written, while completing `update_ekf()` is left for assignment. See below.
* `tools.py` provides a function `Jacobian()` to calculate the Jacobian matrix needed in our filter's update algorithm.
*  `plot.py` creates a 2D plot comparing the ground truth against our estimation. The following figure illustrates an example:

![Testing of EKF with Sensor Fusion][EKF-results]

### Assignment

Complete the implementation of EKF with sensor fusion by writing the function `update_ekf()` in the module `kalman_filter`. Details are given in class and instructions are included in comments.
--------------------------------------------

---
칼만필터는 입력과 출력이 하나씩인 아주 간단한 구조로 측정값이 입력되면 내부에서 처리한다음 다음 추정값을 출력한다.
먼저 추정값과 오차 공분산을 예측한다.
1. X2k = AX1k-1
2. P1k = AP2k-1A.transpose + Q 

3. 칼만이득계산을한다.
4. 추정값 계산을 한다.
5. 오차 공분산 계산을 한다. 
이과정을 반복하며 측정값과 추정값을 reculsive 하게 반복하여 찾아낸다.

한편 먼저 칼만필터는 3가지의 조건을 만족해야 성립한다.
1. markov asumption (state complete 해야한다. - 정답이 있다면 반드시 찾을것이고, 없다면 반드시 못찾는다.
2. 선형 state trasition probability 해야한다.
3. 선형 measurement probability 해야한다.

따라서 선형일수가 없는 실제가 많으니 extend 칼만필터를 사용한다. 선형일 필요가 없다.
이때에는 칼만필터의 선형식을 이용할때 자코비안 행렬을 사용한다. (선형적이지 않은 미분한 행렬값을 사용하기 위해서)
미분할때 자코비안 행렬을 이용한다 이외에는 칼만 필터와 같다

# 1. Compute Jacobian Matrix H_j
        H_j = Jacobian(self.x)
# x에 대한 미분한 자코비안행렬값을 구한후 
        
        # 2. Calculate S = H_j * P' * H_j^T + R
# 칼만 게인에 대한 값을 구하기 위해 S 값을 계산한다.
        S = np.dot(np.dot(H_j, self.P), H_j.T) + self.R
        # 3. Calculate Kalman gain K = H_j * P' * Hj^T + R
        K = np.dot(np.dot(self.P, H_j.T), np.linalg.inv(S))
# 그후 거리를 계산한다.        
        # 4. Estimate y = z - h(x')
        px, py, vx, vy = self.x
        rho_est = sqrt(px ** 2 + py ** 2)
        phi_est = atan2(py, px)
        rho_dot_est = (px * vx + py * vy) / sqrt(px ** 2 + py ** 2)
        y = z - np.array([rho_est, phi_est, rho_dot_est], dtype=np.float32)
# 노멀라이즈를 한다.
        # 5. Normalize phi so that it is between -PI and +PI
        phi = y[1]
        while phi > pi:
            phi -= 2 * pi
        while phi < -pi:
            phi += 2 * pi
        y[1] = phi
# 칼만게인을 갱신하기 위해 새로운 계산을 한다.
        # 6. Calculate new estimates
        #    x = x' + K * y
        #    P = (I - K * H_j) * P
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H_j), self.P)
