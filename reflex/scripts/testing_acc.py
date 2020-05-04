#!/usr/bin/env python

import sys
import numpy as np
import scipy
import threading
import time
import rospy
from std_srvs.srv import Empty, Trigger, TriggerResponse
from std_msgs.msg import Bool
from bwhpf_ex import butter_bandpass_filter
from readings2 import HandReader, _NUM_FING, _NUM_SENSORS, _HAND_VERSION, _NUM_SAMPLES, _DIFF, PoseCommand, VelocityCommand, ForceCommand, Hand, Finger
from threading import Thread
from reflex_grasp import ReflexInterface
import matplotlib.pyplot as plt
import matplotlib.animation as animation

norm = True

rospy.init_node('hand_acc', anonymous=True)

reader = HandReader(independent=False)
reader.collect()  # THIS IS KEY - w/o this the reader won't do any filtering operations

# Parameters
x_len = 200         # Number of points to display
y_range = [-0.03, 0.03]
y_range_norm = [0, 0.05]

# Create figure for plotting
fig = plt.figure()
axx = fig.add_subplot(4, 1, 1)
axy = fig.add_subplot(4, 1, 2)
axz = fig.add_subplot(4, 1, 3)
ax  = fig.add_subplot(4, 1, 4)

xs = list(range(0, 200))

ys = [0] * x_len
ysfilt = [0] * x_len
ysx = [0] * x_len
ysy = [0] * x_len
ysz = [0] * x_len
ysxfilt = [0] * x_len
ysyfilt = [0] * x_len
yszfilt = [0] * x_len
axx.set_ylim(y_range)
axy.set_ylim(y_range)
axz.set_ylim(y_range)
ax.set_ylim(y_range_norm)


# Create a blank line. We will update the line in animate
line, = ax.plot(xs, ys, 'b-', label='Acc Norm')
linefilt, = ax.plot(xs, ysfilt, 'r--', label='Norm of filtered accelerations')
linex, = axx.plot(xs, ysx, 'b-', label='Acceleration X')
liney, = axy.plot(xs, ysy, 'b-', label='Acceleration Y')
linez, = axz.plot(xs, ysz, 'b-', label='Acceleration Z')
linexfilt, = axx.plot(xs, ysxfilt, 'r--', label='w/ Highpass Filter')
lineyfilt, = axy.plot(xs, ysyfilt, 'r--', label='w/ Highpass Filter')
linezfilt, = axz.plot(xs, yszfilt, 'r--', label='w/ Highpass Filter')

# Add labels
plt.title('Norm Acceleration Over Time')
plt.xlabel('Samples')
plt.ylabel('Acceleration Norm (m/s^2)')

# This function is called periodically from FuncAnimation
def animate(i, ys, ysfilt, ysx, ysxfilt, ysy, ysyfilt, ysz, yszfilt):

    # Get readings
    acc = reader.get_acc()
    filt = reader.get_acc_hifilt()
    acc_norm = reader.get_acc_norm()
    filt_norm = reader.get_acc_norm_hifilt()

    # Add readings to data
    ys.append(acc_norm)
    ysfilt.append(filt_norm)
    ysx.append(acc[0])
    ysy.append(acc[1])
    ysz.append(acc[2])
    ysxfilt.append(filt[0])
    ysyfilt.append(filt[1])
    yszfilt.append(filt[2])

    # Limit y list to set number of items
    ys = ys[-x_len:]
    ysfilt = ysfilt[-x_len:]
    ysx = ysx[-x_len:]
    ysy = ysy[-x_len:]
    ysz = ysz[-x_len:]
    ysxfilt = ysxfilt[-x_len:]
    ysyfilt = ysyfilt[-x_len:]
    yszfilt = yszfilt[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)
    linefilt.set_ydata(ysfilt)
    linex.set_ydata(ysx)
    linexfilt.set_ydata(ysxfilt)
    liney.set_ydata(ysy)
    lineyfilt.set_ydata(ysyfilt)
    linez.set_ydata(ysz)
    linezfilt.set_ydata(yszfilt)

    return line, linefilt, linex, linexfilt, liney, lineyfilt, linez, linezfilt,

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(ys,ysfilt,ysx,ysxfilt,ysy,ysyfilt,ysz,yszfilt,),
    interval=50,
    blit=True)
plt.show()
