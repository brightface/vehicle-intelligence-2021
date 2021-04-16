# Week 4 - Motion Model & Particle Filters

---

[//]: # (Image References)
[empty-update]: ./empty-update.gif
[example]: ./example.gif

## Assignment

You will complete the implementation of a simple particle filter by writing the following two methods of `class ParticleFilter` defined in `particle_filter.py`:

* `update_weights()`: For each particle in the sample set, calculate the probability of the set of observations based on a multi-variate Gaussian distribution.
* `resample()`: Reconstruct the set of particles that capture the posterior belief distribution by drawing samples according to the weights.

To run the program (which generates a 2D plot), execute the following command:

```
$ python run.py
```

Without any modification to the code, you will see a resulting plot like the one below:

![Particle Filter without Proper Update & Resample][empty-update]

while a reasonable implementation of the above mentioned methods (assignments) will give you something like

![Particle Filter Example][example]

Carefully read comments in the two method bodies and write Python code that does the job.

---
파티클필터
1. 파티클 필터는 확장된 칼만필터와 같은 비선형 시스템 모델을 갖는다.
2. 각 파티클을 통한 ukF의 시그마 포인트와 유사한 파티클이라는 다수의 데이터를 사용한다.
3. 정규분포를 따르지 않는 시스템 모델이지만 여기에서는 2개의  가우시안 모델을 따른다고 한다.

파티클 알고리즘
0. 파티클초기화
1. 파티클 예측
2. 파티클 가중치갱신
3. 추정값계산
4. 재샘플링

1번부터 4번까지 반복하여 측정값을 통한 추정값을 계산해 낸다.

시스템 모델 : 수평거리, 이동속도, 고도 (X1, X2, X3)
Z = np.squrt(X1**2 + X2**2) + v
  == h(x) + v
  
   # 1. Select the set of landmarks that are visible (within the sensor range).
        for p in self.particles:
            visible_landmarks = []
            for id, landmark in map_landmarks.items():
                if distance(p, landmark) <= sensor_range:
                    landmark['id'] = id
                    visible_landmarks.append(landmark)

   # 2. Transform each observed landmark's coordinates from theparticle's coordinate system to the map's coordinates.
            transformed_observations = []
            for obs in observations:
                obs['x'] = obs['x'] + p['x'] * np.cos(p['t']) - \
                                      p['y'] * np.sin(p['t'])
                obs['y'] = obs['y'] + p['x'] * np.sin(p['t']) + \
                                      p['y'] * np.cos(p['t'])
                transformed_observations.append(obs)

   # 3. Associate each transformed observation to one of the predicted (selected in Step 1) landmark positions.
        #    Use self.associate() for this purpose - it receives
        #    the predicted landmarks and observations; and returns
        #    the list of landmarks by implementing the nearest-neighbour
        #    association algorithm.
            assoc_landmarks = self.associate(visible_landmarks, transformed_observations)
            p['accoc'] = [landmark['id'] for landmark in assoc_landmarks]

   # 4. Calculate probability of this set of observations based on
        #    a multi-variate Gaussian distribution (two variables being
        #    the x and y positions with means from associated positions
        #    and variances from std_landmark_x and std_landmark_y).
        #    The resulting probability is the product of probabilities
        #    for all the observations.
            particle_loc = np.array([p['x'], p['y']])
            landmark_x = np.array([landmark['x'] for landmark in assoc_landmarks])
            landmark_y = np.array([landmark['y'] for landmark in assoc_landmarks])
            print(landmark_x)
            print(landmark_y)
            mean_x = landmark_x.mean(axis=0)
            mean_y = landmark_y.mean(axis=0)
            mean = np.array([mean_x, mean_y])
            
            corr = np.correlate(landmark_x, landmark_y)[0]

            cov = np.array([[std_landmark_x**2, corr*std_landmark_x*std_landmark_y],
                            [corr*std_landmark_x*std_landmark_y, std_landmark_y**2]])
            particle_prob = multivariate_normal.pdf(particle_loc, mean=mean, cov=cov)
   # 5. Update the particle's weight by the calculated probability.
            p['w'] = particle_prob

   # Resample particles with replacement with probability proportional to their weights.
    def resample(self):
        # TODO: Select (possibly with duplicates) the set of particles
        #       that captures the posteior belief distribution, by
        # 1. Drawing particle samples according to their weights.
        weights = [p['w'] for p in self.particles]
        new_particles = np.random.choice(self.particles, self.num_particles, p=weights)

        # 2. Make a copy of the particle; otherwise the duplicate particles
        #    will not behave independently from each other - they are
        #    references to mutable objects in Python.
        #    Finally, self.particles shall contain the newly drawn set of
        #    particles.
        self.particles = []
        for new in new_particles:
            tmp_particle = new
            self.particles.append(tmp_particle)

        return
