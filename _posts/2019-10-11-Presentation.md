---
title: Vibration Reduction of Flexible Beam with Bayesian Input shaper
classes: wide
--- 

```python
#Imports
from IPython.core.display import Image, display
from IPython.core.display import HTML

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams["figure.figsize"] = (20,10)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

import json
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
%matplotlib inline
#
```

## Introduction

I work in the CERN Robotic Team as a PhD student. One of the experiments has asked us to design a mobile robot for inspection of their facilities. 
The robot has to fit 2 main criteria:

* has to be able to pass under 20 cm obstacles
* must be able to perform visual inspection between 20 cm and 1.5 meters

To ensure these requirements are met, we decided to fit a pan-tilt camera on top of a flexible beam.    
The beam can be raised with a motor up to 2 meters high. It can also be lowered to the base of the robot.

![]({{ site.baseurl }}/assets/images/Presentation/cernbot_small.jpg)

The beam has a low weight so it can be moved quickly.
This however means that the beam is not stiff. It vibrates when the robot is moving. This results in poor video feed.
The vibration is induced by the linear acceleration of the robot and the unevenness of the surface.

Bellow you can see the video taken on top of the flexible beam.
It can be observed that the oscillation of the camerea, is mainly caused by the initial acceleration of the robot.
It dies down very quickly when it is moving with constant velocity tough.


```python
HTML('<iframe width="300" height="500" src="https://www.youtube.com/embed/1AbahaqgWqU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

    /home/robotronics/.local/lib/python3.5/site-packages/IPython/core/display.py:694: UserWarning: Consider using IPython.display.IFrame instead
      warnings.warn("Consider using IPython.display.IFrame instead")





<iframe width="300" height="500" src="https://www.youtube.com/embed/1AbahaqgWqU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



## Brain Storming of Solutions

To get rid of the vibration of the camera, I can see 4 possible solutions.

### 1. Drive slowly
* <span style="color:red">limits the movement of the robot</span>
* <span style="color:green">no need to implement anything</span> 

### 2. Balance with accelerating/breaking the robot
* <span style="color:red">needs a sensor to give feedback of the beam position</span>
* <span style="color:red">needs perfect control of the robotic base</span>
* <span style="color:red">the beam movement must be modeled</span>
* <span style="color:green">no need for additional actuator</span>
* <span style="color:green">easy to implement the hardware, medium hard software</span>

### 3. Corrigate the video camera with a gimbal
* <span style="color:red">needs a sensor to give feedback of the beam position</span>
* <span style="color:red">needs 2 additional gimbal motors</span>
* <span style="color:red">an additonal software needed for the control of the gimbal</span>
* <span style="color:green">no need to change the code of the robot base</span>
* <span style="color:green">medium hard hardware, medium hard software</span>

### 4. Use reaction wheels to stabilize the top of the beam
* <span style="color:red">needs a sensor to give feedback of the beam position</span>
* <span style="color:red">needs 2 custom machined reaction wheels (very expensive)</span>
* <span style="color:red">needs 2 additional motors to rotate the wheels</span>
* <span style="color:red">complicated control to stabilize the beam</span>
* <span style="color:red">the high weight makes the beam movement slower</span>
* <span style="color:green">no need to change the code of the robot base</span>
* <span style="color:green">very hard hardware, very hard software</span>

I decided to go with solution number 2. The main reason is that it doesn't require additional hardware components, just a simple sensor. 

In my experience in a system the bigger the complexity of the hardware the higher the cahnce to fail during extended operation.

## Experiment Set UP


### The Hardware

The robot base is equipped with 4 Maxon EC motors. The motors are controlled with the EPOS2 servo controller.
The brain of the robot is a commercially available NUC computer with Ubuntu on it. 
To get the position feedback from the beam I have placed a VMU931 IMU sensor on the top. 
The sensor has built in accelerometer, gyroscope and magnetometer. 
I only used the gyroscope measurements, because they have lower noise compared to the accelerometer.
Since we do not yet have the camera, I have placed 420 g weight (biggest wrench at hand) on top of the beam to emulate it.

### The Software

The control of the robot is handled by the CERN Robotic Framework (CRF).
This is a framework developed by the members of the CERN Robotics team.
It is written in C++, runs on Ubuntu.

For my application I needed to be able to control the movement of the robot precisely.
I have modified our interface to the motor drivers to be able to set both the velocity and acceleration of the motors.
Reviewed the control loop of the 4 robots. Originally they needed on average 20 ms to change the state of the 4 robots. 
I managed to optimize the code, and cut it down to an average of 6 ms. Additionally the motor movements are synchronized to a 2 ms window. This makes the robot movement much smoother, and also there is less torque ripple between the 4 motors.   

The motors of the robot are controlled by a current PI loop and a PI velocity loop with an additional feedforward term. 
The controllers were originally tuned by the Maxon software, while the motors were freerunning. 
This meant that each motor had slightly different parameters. Their movement was slightly different.
Since the robot has a considerable weight the controller was undertuned, it was unable to reach the desired speed and acceleration. 
I manually retuned them, to have a better performance. 


```python
# Plotting the velocity curves
motor_bad = genfromtxt("robotBaseLog_bad.csv", delimiter=";")
motor_good = genfromtxt("robotBaseLog_good.csv", delimiter=";")

fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
ax0.set_xlabel('Time [msec]', size=15)
ax0.set_ylabel('Motor Velocity [relvalue]', size=15)

time0 = motor_bad[:,0] - motor_bad[0,0]
time0 = np.rint(time0 / 1e6)
ax0.plot(time0, motor_bad[:,1:4])
ax0.set_xlim([min(time0), max(time0)])
ax0.set_ylim([0,16])
ax0.grid()
ax0.set_title("Automatically tuned controller")

time1 = motor_good[:,0] - motor_good[0,0]
time1 = np.rint(time1 / 1e6)
ax1.plot(time1, motor_good[:,1:4])
ax1.set_xlim([min(time1), max(time1)])
ax1.set_ylim([0,16])
ax1.grid()
ax1.set_xlabel('Time [msec]', size=15)
ax1.set_title("Manually tuned controller")

fig.set_size_inches(12, 6, forward=True)

plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_6_0.png)


Above you can see the velocities of each motor before and after tuning the controllers. 

The VMU931 communicates with the computer through the serial port. 
The CRF already contained an interface for it.
I rewrote major part of it, to make it more robust and increase the signal acquisition frequency. 

Finally to make experiments and the plotting of results easier, I wrote a python wrapper for the code with Boost.Python. 
This allows the user to control the robot speed and acceleration from Jupyter Notebook.
The notebook recieves the motor torques and velocities, the gyroscope measurements and the robot position.
This data can quickly be plotted with Matplotlib in the notebook.

## Input shaping

Input shaping is an open-loop control method, which strives to reduce the vibration of flexible mechanisms.    
It is mainly used in cranes and flexible link robotic arms in practice.    
Input shaping can be used to control linear asymptotically stable flexible systems.
The main idea of input shaping lies in the superposition principle.    
The flexible system starts vibrating when an acceleration is introduced to this system, which resuslts in the vibration around zero position.    
When accelerating the structure, instead of using a single impulse, several impulses are applied.    
If the impulses are applied out of phase of the vibration, they simply cancel each other out.    

Let us look at a second order harmonic oscillator. The physical realization of such systems is for example a pendulum.

Bellow the speed of the pendulum can be seen related to time.

\begin{align}
y_{o}(t) & = \left[ \frac{A_{0} \omega}{\sqrt{1 - \xi^2}} e^{-\xi \omega \left( t - t_{0} \right)} \right] sin\left( \omega \sqrt{1 - \xi^2} \left( t - t_{0} \right) \right)
\end{align}

\begin{align}
\omega_{d} & = \omega \sqrt{1 - \xi^2}
\end{align}


```python
# Second order harmonic oscillator impulse response
def dampedVibration(x, A0, omega, damping, x0):
    if (damping >1):
        return np.inf
    coeff = np.sqrt(1 - damping*damping)
    y = ((A0 * omega) / coeff) * np.exp(-damping * omega * (x - x0)) * np.sin(omega * coeff* (x - x0))
    return y

x = np.linspace(0, 10, 200)
y = dampedVibration(x, 7, 2*np.pi*0.5, 0.15, 0)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(x, y)
ax.set_xlim([0, 10])
ax.set_ylim([-20,20])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Velocity [1]', size=15)
ax.set_title("Under Damped Oscillation")

plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_9_0.png)


If we have several impulses we can sum the effects together, since the system is linear.

\begin{align}
y_{\sum} \left( t \right) & = \sum_{i=1}^{n}\left[ \frac{A_{0} \omega}{\sqrt{1 - \xi^2}} e^{-\xi \omega \left( t - t_{0} \right)} \right] sin\left( \omega_{d} \left( t - t_{0} \right) \right)
\end{align}




```python
# Superposition of impulses
x = np.linspace(0, 10, 200)
y1 = dampedVibration(x, 7, 2*np.pi*0.5, 0.15, 0)
y2 = -dampedVibration(x, 7, 2*np.pi*0.5, 0.15, 0)
y2[:int(1/10*200)] *= 0
y3 = y1 + y2
fig, ax = plt.subplots(figsize=(10,5))
p1 = ax.plot(x, y1)
p2 = ax.plot(x, y2)
p3 = ax.plot(x, y3, linewidth=4.0)
ax.set_xlim([0, 10])
ax.set_ylim([-20,20])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Position [1]', size=15)
ax.legend((p1[0],p2[0],p3[0]),
          ("1st impulse response","2nd impulse response","Superpoisiton of the two"),
            prop={'size': 13})
ax.set_title("Superposition of impulses")
plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_11_0.png)



```python
# Superposition of impulses
x = np.linspace(0, 10, 200)
y1 = dampedVibration(x, 7, 2*np.pi*0.5, 0.15, 0)
y2 = -dampedVibration(x, 7, 2*np.pi*0.5, 0.15, 0)
y2[:int(1/10*200)] *= 0
y3 = y1 + y2
fig, ax = plt.subplots(figsize=(10,5))
p1 = ax.plot(x, y1)
p2 = ax.plot(x, y2)
p3 = ax.plot(x, y3, linewidth=4.0)
ax.set_xlim([0, 10])
ax.set_ylim([-20,20])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Position of the Camera', size=15)
ax.legend((p1[0],p2[0],p3[0]),
          ("Normal Movement","Counter Balancing","The result of the two together"),
            prop={'size': 13})
ax.set_title("How to get rid of the camera vibration?")
plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_12_0.png)


As can be seen on the figure above the method gets rid of the oscillation in the ideal case.    
Unfortunately, there is still a bit of oscillation, since the first half wave is not eliminated.

The method has 3 sets of parameters: number of impulses, the time location of impulses and the proportion of the impulses.    
In the simplest case there is only a pair of impulses, this is called Zero Vibration shaper.

Bellow you can see how to calculate the amplitudes ($A_{i}$) and the time position ($T_i$) of the controller.
$$
\begin{bmatrix}
    A_{i}        \\
    t_{i}       
\end{bmatrix} =
\begin{bmatrix}
    \frac{1}{1 + K}  & \frac{K}{1 + K}     \\
    0  & \frac{\pi}{\omega \sqrt{1 - \xi^{2}}}   
\end{bmatrix}
$$

$$
K = \exp^{\left( \frac{- \xi \pi}{\sqrt{1 - \xi^{2}}} \right)}
$$

For the method to work only the natural frequency and the damping factor of the beam is needed. There is no need for complicated models.

Normally the beam equations are modelled by partial differential equation , either by Bernouilli or Timoshenko. These equations are hard to solve, and their precision highly depends on the precision of the initial conditions.    

I applied an impulse to the flexible beam and recorded the angular velocity of the tip with the gyroscope.
Then fitted the second order harmonic scillator equation to the beam velocity data.    
As can be seen the plot bellow, the model fits the data pretty well.    
Unfortunately, there is a phase shift in the later periods. This might be because the data gets noisy around the peaks, for the first few periods. Since the goodness of the fit is mostly influenced by the first few oscillations.


```python
# Plot of rod natural frequency
gyroscope = genfromtxt("vmulog_free.csv", delimiter=";")
y = gyroscope[785:,1]
x = np.linspace(0, y.shape[0] * 0.005, y.shape[0])
fig, ax = plt.subplots(figsize=(10,5))
p1  = ax.plot(x, y)

p0 = [30, 1, 0.1, 0]
popt, pcov = curve_fit(dampedVibration, x, y, p0)
p2 = ax.plot(x, dampedVibration(x, popt[0], popt[1], popt[2], popt[3]))
ax.set_xlim([0, max(x)])
ax.set_ylim([min(y) * 1.2 ,max(y)*1.2])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Velocity [1]', size=15)
ax.legend((p1[0],p2[0]), ("Gyroscope Data","Secod order fit"), prop={'size': 15})
ax.set_title("Natural Frequency of the rod")

ax.text(0.9, 0.2,'$\\omega$ = {:.2f} \n $\\xi$ = {:.2f}'.format( popt[1], popt[2]),
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, size = 20)
plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_14_0.png)



```python
# Calculate K and T1
damping = popt[2]
K = np.exp((-damping * np.pi) / (1- damping* damping))
print("The K factor is : {:.2f}".format(K))
print("The first impulse is {:.2f}%, the second impulse is {:.2f}% of the acceleration.".format(1/(1+K)*100, K/(1+K)*100))
omega = popt[1]
# This is in VMU measurement units, which is 5 millisec
t_i = np.pi / (omega * np.sqrt(1-damping* damping))
print("The second impulse is at : {:.3f} msec".format(t_i))
#
```

    The K factor is : 0.59
    The first impulse is 62.82%, the second impulse is 37.18% of the acceleration.
    The second impulse is at : 0.858 msec


There is one big problem with this method of vibration reduction.
Input shaping gives no guidelines for evaluating how well the shaper works.
It does not take into account the real system. The shaper is ideal only if the actuators and controllers of the robot can produce the required impulses at just the right time.
In real world however there is always some timedelays and overshoots in the controllers.
In my case since I was working with a heavy mobile robot, which has 4 motors I wanted to know if I could do better.

# Cost Functions

One way to get a better input shaper is to try different parameters on the real robot and determine which one is better.
From this feedback we could create an optimization/learning algorithm to find the optimal parameters.

The first step is to create a metric which will show if one run of the parameters were better than the other. I will call this metric loss function.
The loss function has to satisfy the following:
1. the value should not vary between consecutive trials with same parameters
2. the value should change between different sets of parameters

I decided to use the root-mean-square error (RMSE) of the gyroscope values. For the mean I have substitued zero, since the values should be 0 in the longer constant speeds. (The mean of the measurement would be very different from zero, since the oscillation is not symmetric.)

$
RMSE = \sqrt{\frac{\sum{ \left( y_{i} - 0 \right)^{2}}}{N}}
$

Since the robot wheels have no suspensions, the uneven ground will introduce noise to the gyroscope signal.
It is important to get rido of this, otherwise the loss function will be very noisy.

I have recorded a longer run of the robot with constant speed. In constant speed after the inital period the gyroscope should measure values around zero. 

In the figure bellow the measurement values can be seen. As expected the measurements are centered around zero, and they have symmetrical stochastic noise.    


```python
#Steady state
gyroscope = genfromtxt("vmulog_constantv.csv", delimiter=";")
y = gyroscope[:,1]
x = np.linspace(0, y.shape[0] * 0.005, y.shape[0])
fig, ax = plt.subplots(figsize=(15,5))
p1  = ax.plot(x, y)
ax.set_xlim([0, max(x)])
ax.set_ylim([min(y) * 1.2 ,max(y)*1.2])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Velocity [1]', size=15)
ax.set_title("Gyroscope readings during constant speed movement")

plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_19_0.png)


To better see the noise spectrum I have also plotted the fast fourrier transformation (FFT) of the signal.
There are 3 peaks in the signal. From the previous experiment it can be seen that the beam's natural frequency is around 1 Hz. We can apply a Digital Butterworth filter to get rid of the components higher than 4-5 Hz, to make the loss calculations less noisy.

Bellow the filtered and the original signal can be seen right after the acceleration phase.


```python
# Fourrier Transform of steady state
import scipy.fftpack
T = 0.005 # sampling time period
N = y.shape[0]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax.set_xlim([0, max(xf)])
ax.set_ylim([min(2.0/N * np.abs(yf[:N//2])) * 1.2 ,max(2.0/N * np.abs(yf[:N//2]))*1.2])
ax.grid()
ax.set_xlabel('Frequency [Hz]', size=15)
ax.set_ylabel('Amplitude [rad/sec]', size=15)
ax.set_title("Fourrier Transformation of the Gyroscope signal")
plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_21_0.png)



```python
#Filtered Signal
y = genfromtxt("vmulog_accel.csv", delimiter=";")[:,1]
x = np.linspace(0, y.shape[0] * 0.005, y.shape[0])
from scipy import signal
fs = 200 #sampling freq
fc = 4
w = fc / (fs / 2)
b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, y)

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x, y, marker = ".", alpha = 0.3)
p0 = ax.plot(x, y, marker = ".", alpha = 0.3)
p1 = ax.plot(x, output, linewidth=4)
ax.set_xlim([0, max(x)])
ax.set_ylim([min(y) * 1.2 ,max(y)*1.2])
ax.grid()
ax.set_xlabel('Time [sec]', size=15)
ax.set_ylabel('Velocity [1]', size=15)
ax.set_title("Comparison of the raw and filtered measurements")
ax.legend((p0[0],p1[0]), ("Gyroscope Data","Filtered Data"), prop={'size': 15})

plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_22_0.png)


# Bayesian Optimization

Now that we have a metric to evaluate which control parameters are better we need to choose an algorithm to propose new parameters.

Machine learning in robotics have several difficulties compared to computer vision:
- there are a limited number of robots, learning can not be parallelized
- each control function evaluation takes long time
- real world robot control and sensors are very noisy

On the other hand when we do not use computer vision, just low dimensional sensor data, the models are much simpler. A gyroscope and the 4 wheels represent 7 dimensions. A small resolution image with 64 x 64 x 3 pixels represents 12288 dimensions.

The available algorithms are also constrained because we have no access to the gradient of the control parameters. 

After careful consideration I have chosen Bayesian optimization for this problem. This can be also considered a reinforcement learning algorithm. 

Bayesian optimization is a global optimization method. Theoretically it finds the best set of parameters.
It does not need the gradients of the loss function. Works with relatively small amount of data.

It is mostly used for finding the optimum in cases where the cost function is very expensive to evaluate. Such as predicting which point is ideal for oil drilling (Krieging), or optimizing machine learning model hyperparameters.

Bayesian optimization works by creating a model of the loss surface based on the previously evaluated points, than proposing a new set of parameters to evaluate.

This loss surface is an $\mathbb{R}^{n} \rightarrow  \mathbb{R}$ function, where n is the number of model parameters. The most common model to use are Gaussian Processes. Other common choice is using decision trees.

Gaussian Processes (GP) are probabilistic curves, where the finite collection of points form a multivariate normal distribution. To put it simpler this is just a different way to fit a curve, can be compared to line fitting with OLS. Instead of fitting a parametric model, such as a line or a sine wave, the function value is determined by key points, and a kernel function which shows how the function value changes between the key points.

Unlike parametric models GPs model the function in a discrete interval. When using Bayesian Optimization we have to set the range where we are looking for the parameter.

GPs are probabilistic models. They do not represent a single function, but a collection of stochastic functions. In practice we look at the mean, most likely, values of the function. From the standard deviaton of these functions we can calculate how certain is a Gaussian process in the function value in a given point. As a rule of thumb the further away we are from the key points, the less certain the model is.



```python
# Fitting a Gaussian Process to the Sine function

def f(x):
    return x * np.sin(x)

x = np.linspace(0,10, 100)
y = f(x)

number_of_samples = 12
gp =GaussianProcessRegressor(
            kernel=Matern(),
            normalize_y=True,
            random_state=5
        )
x_star = np.random.uniform(min(x), max(x), number_of_samples)

y_star = f(x_star)
gp.fit(x_star.reshape(-1, 1), y_star)

m, v = gp.predict(x.reshape(-1, 1), return_std=True)

plt.figure(figsize=(15,8))

p0 = plt.plot(x,y)
fill = plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([m - 1.9600 * v,
                        (m + 1.9600 * v)[::-1]]),
         alpha=.1, fc='b', ec='None', label='95% confidence interval')
sc = plt.scatter(x_star, y_star)
p1 = plt.plot(x,m)

plt.legend((p0[0],p1[0], fill[0], sc),
           ("f(x)","Gaussian Process Mean","95% confidence interval", "Random Samples" ),
           prop={'size': 15})
plt.title("Fitting a Gaussian Process to y = x * sin(x)")
plt.show()
#
```


![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_26_0.png)


Based on the model of the loss at each iteration we need to pick a new set of parameters to probe.
Acquisition function is responsible to suggesting the new location.

There are three common acquisition functions : Expected Improvement (EI), Upper Confidence Bound (UCB) and Probabilty of Improvement (POI).

They suggest new points based on the mean and the standard deviation of the loss. Usually they have a parameter where we can choose the trade off between exploration and explotiation.

If we prefere exploration the acquisition function will prefere points in sample space, where the standard deviation is high. This means that the model is uncertain about the loss values in the given point. By exploring the space better, we ensure that we will find the global imptimum of the problem. On the other hand the optimization will take longer.

If we focus on exploitiation, then the new parameters will be close to points where the mean is minimal. It will make the optimization converge faster to a minumum. However since the algoirthm evaluates smaller part of the parameter space it risks getting stuck in a local optimum.

The trade of between the two have to be set manually. It must be based on how much time we have to evaluate the algorithm.

# Experiments

For the Bayesian Optimization I have tried out several libraries.
I chose to use fmfn/BayesianOptimization because it has very nice logging functionalities.

I run several trials with Matern and RBF kernels of the GaussianProcess.
Tried out the 3 previously mentioned acquisition functions.

The goal of the optimization was to find the optimal shaper parameters. 
To have as much vibration as possible I have fixed the first acceleratioin impulse close to the maximum acceleration of the robot. This way only the time location of the second impulse ($t_{2}$), and the amplitude ($A_{2}$) of the second impulse had to be found. 

The robot was running in the lab. There are 2 bumpers in the runway, which add noise to the trials.
To make each execution as similar as possible, I did one run forward and one backward. Calculated the mean of the two runs. Each trial took around 15 seconds. An experiment consisted of 5 initial random values, and 30 optimization steps.

Bellow you can see a video of one optimization process. The posterior mean, corresponds to the predicted value of the loss function. The higher the value, the better the parameters are. The posterior standard deviation shows the uncertainty of the predicted mean. It can be observed that at points which were evaluated the standard deviation is close to zero. At points far from evaluations the standard deviation is higher.


```python
# Optimization video
gp =GaussianProcessRegressor(
            kernel=Matern(length_scale=[5, 10]),#RBF(length_scale=[0.05, 1]),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=5
        )
#Load data
log = "logs_kappa5.json"
x = []
y = []
with open(log, "r") as f:
    for line in f:
        line = json.loads(line)
        y.append(line["target"])
        x.append([line["params"]["a1"],line["params"]["t1"]])
x = np.array(x)

label_x = "A1"
label_y = "T1"
bounds = [[0, 1.5],[300,1000]]

X1 = np.linspace(0, 1.5, 200)
X2 = np.linspace(300, 1000, 200)
x1, x2 = np.meshgrid(X1, X2)
X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))

fig = plt.figure(figsize=(13,5))


def update(i):
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    gp.fit(x[:i],y[:i])
    m, v = gp.predict(X, return_std=True)

    cf1 = ax1.contourf(X1, X2, m.reshape(200,200),100)
    ax1.plot(x[:i-1,0], x[:i-1,1], 'r.', markersize=10, label=u'Observations')
    ax1.plot(x[i,0], x[i,1], 'r.', markersize=25, label=u'New Point')
    cb1 = fig.colorbar(cf1, ax=ax1)
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    
    ax1.set_title('Posterior mean')
    ##
    ax2.plot(x[i,0], x[i,1], 'r.', markersize=25, label=u'New Point')
    ax2.plot(x[:i-1,0], x[:i-1,1], 'r.', markersize=10, label=u'Observations')
    cf2 = ax2.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
    cb2 = fig.colorbar(cf2, ax=ax2)
    ax2.set_xlabel(label_x)
    ax2.set_ylabel(label_y)
    
    ax2.set_title('Posterior sd.')
    return ax1, ax2
#
```


    <Figure size 936x360 with 0 Axes>



```python
anim = FuncAnimation(fig, update, frames=np.arange(3, x.shape[0]), interval=500)
anim.save('line.gif', dpi=80, writer='imagemagick')
HTML(anim.to_html5_video())
```




<video width="936" height="360" controls autoplay loop>
  <source type="video/mp4" src="data:video/mp4;base64,AAAAHGZ0eXBNNFYgAAACAGlzb21pc28yYXZjMQAAAAhmcmVlAAJNRG1kYXQAAAKuBgX//6rcRem9
5tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTU1IHIyOTE3IDBhODRkOTggLSBILjI2NC9NUEVHLTQg
QVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDE4IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcv
eDI2NC5odG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9
MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0xLjAwOjAuMDAgbWl4ZWRfcmVm
PTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6
b25lPTIxLDExIGZhc3RfcHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MTEgbG9v
a2FoZWFkX3RocmVhZHM9MSBzbGljZWRfdGhyZWFkcz0wIG5yPTAgZGVjaW1hdGU9MSBpbnRlcmxh
Y2VkPTAgYmx1cmF5X2NvbXBhdD0wIGNvbnN0cmFpbmVkX2ludHJhPTAgYmZyYW1lcz0zIGJfcHly
YW1pZD0yIGJfYWRhcHQ9MSBiX2JpYXM9MCBkaXJlY3Q9MSB3ZWlnaHRiPTEgb3Blbl9nb3A9MCB3
ZWlnaHRwPTIga2V5aW50PTI1MCBrZXlpbnRfbWluPTIgc2NlbmVjdXQ9NDAgaW50cmFfcmVmcmVz
aD0wIHJjX2xvb2thaGVhZD00MCByYz1jcmYgbWJ0cmVlPTEgY3JmPTIzLjAgcWNvbXA9MC42MCBx
cG1pbj0wIHFwbWF4PTY5IHFwc3RlcD00IGlwX3JhdGlvPTEuNDAgYXE9MToxLjAwAIAAAEEGZYiE
ABX//vfJ78Cm61tbtb+Tz0j8LLc+wio/blsTtOoAAAMAAAMAAAMAAAMB9kEC0xAADmDtP+Ehj4AK
OHaBUznIcsUF0EgKbMPq3ztPZjth9suD7SHOgxa5nOACSj4OShurmPqSKnYMLaPnGrqaXos8dn7E
QBE40rTGMDW+Ic6+IBjav4gzbnPpnEaMr2Zphybp3ErZYOTEg0H1RgMipapqPHa4C4H6axQsQCny
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
AAAAAQAAAABMYXZmNTguMjAuMTAw
">
  Your browser does not support the video tag.
</video>



On the figure bellow you can see the comparison of the different shapers. No shaping case is where the robot is left to acclerate from standing position to the desired velocity directly. 

The analytic shaper was created simply using the beam natural frequency and damping. It is clearly superior to the no shaper case.

Finally, the Bayesian Shaper is the result of the optimization method. It has out performed the analytical shaper in all 10 trials.


```python
#Comparison of methods
y = genfromtxt("b_s_n.csv", delimiter=";")
x = np.linspace(1, y.shape[0], y.shape[0])
print(x.shape)
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(x, y[:,0], marker = "X")
ax.scatter(x, y[:,1], marker = "X")
ax.scatter(x, y[:,2], marker = "X")
ax.set_xlim([0, max(x)+1])
ax.set_ylim([5 ,10])
ax.grid()
ax.set_xlabel('Number of Trial', size=15)
ax.set_ylabel('Loss Function (smaller better)', size=15)
plt.legend(("Bayesian Shaper","Analytic Shaper", "No Shaping"))
ax.set_title("Comparison between the methods")
plt.show()
#
```

    (10,)



![alt]({{ site.baseurl }}/assets/images/Presentation/Presentation_33_1.png)


On the video bellow the camera feed can be seen from the top of the beam. 
There is considerably less vibration in the video then during the original control of the robot.


```python
HTML('<iframe width="300" height="500" src="https://www.youtube.com/embed/lYZm8Lcw7tc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

    /home/robotronics/.local/lib/python3.5/site-packages/IPython/core/display.py:694: UserWarning: Consider using IPython.display.IFrame instead
      warnings.warn("Consider using IPython.display.IFrame instead")





<iframe width="300" height="500" src="https://www.youtube.com/embed/lYZm8Lcw7tc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



# Discussion

In my experience Bayesian Optimization is a very reliable algorithm. It can find the global optima at each trial, and does overperform the analytic case very stably.

The biggest failure points of the method is setting the the initial parameters of the Gaussian Process and finding a good exploration-exploitation trade off. 

The frameworks I have looked at all initialize the Gaussian Process with a single dimensional length scale parameter. If the parameters however are not in the same order of magnitude, the optimization method will disregard one of the dimensions. 

The exploration parameter can be set, by trying a few values. If the optimization quickly converges to local minima, after 5-10 trials, we should increase the exploration term.

In the future this method could be extanded to work with different flexible beam heights. The length of the beam could be considered a parameter, which the algorithm can not change. This type of optimization is called contextual Bayesian Optimization.

Finally, instead of using a 2 impulse shaper Bayesian Optimization could be used to predicta whole velocity trajectory. This would increase the dimensionality of the problem, but I believe it could work very well.
