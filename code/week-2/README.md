# Week 2 - Markov Localization

---

[//]: # (Image References)
[plot]: ./markov.gif

## Assignment

You will complete the implementation of a simple Markov localizer by writing the following two functions in `markov_localizer.py`:

* `motion_model()`: For each possible prior positions, calculate the probability that the vehicle will move to the position specified by `position` given as input.
* `observation_model()`: Given the `observations`, calculate the probability of this measurement being observed using `pseudo_ranges`.

The algorithm is presented and explained in class.

All the other source files (`main.py` and `helper.py`) should be left as they are.


<motion model algorithm>
각스텝마다 확률은 현재에 확률에 이전확률을 곱해서 구한다(구해서 다 이전까지에다 더함).(현재의 확률은 정규분포를 따른다)
따라서 P(Xt | Xt-1) = Ptrans * Pprior 이다.
Xt가 맵 사이즈라 하였을 위치 있을 확률은 맵사이즈 만큼의 조건부 확률은 을 다 더하는것이다.
따라서 for i in range(map_size)  를 돌며 다 position_prob에 다 더하고 
확률은 정규분포를 이전확률에 곱한값을 다 더한다.
단 정규분포의 확률을 구하는 방식은 prob_dist 라서 현재 포지션에 position - 이전포지션 해서 빼줘야한다.





<observation_model algorithm>
이모델은 측정된 observation 의 위치를 가지고 확률 을 계산하는 모델이다.
식은 이전확률에 정규분포확률(관측이 들어감, 예측레인지, 거리)을 다 곱하는것이다. 
ex) observation measurement [1,6]
-probility observation Xt=7
p(Z1t = 1 | Xt = 7, m ) * p(Z2t = 6 | Xt = 7, m ) = 정규분포*정규분포
단 이때 예측 레인지는 observation 길이보다 커야한다.
아닐때는 0을 리턴한다.
If you correctly implement the above functions, you expect to see a plot similar to the following:

![Expected Result of Markov Localization][plot]

If you run the program (`main.py`) without any modification to the code, it will generate only the frame of the above plot because all probabilities returned by `motion_model()` are zero by default.

