#!/usr/bin/env python

import numpy as np
import scipy
import threading
import time
import rospy
from std_srvs.srv import Empty, Trigger, TriggerResponse
from std_msgs.msg import Bool
from reflex_msgs.msg import PoseCommand, VelocityCommand, Hand
from bwhpf_ex import butter_bandpass_filter

_FILTERING = True
_DIFF = 1
_NUM_SAMPLES = 50  # Num samples from 50 Hz on ReFlex TakkTile hand
_FILTER_CUTOFF = 10.0  # Hz

_NUM_FING = 3
_NUM_SENSORS = 9  # per finger
_NUM_DISTALS = _NUM_SENSORS - 5  # distal sensors per finger, true for either version
_PINCER = False  # Use pincer grasp if this is true, power o/w

def butter_highpass(cutoff=_FILTER_CUTOFF, fs=_NUM_SAMPLES, order=1):
    nyq = 0.5 * fs
    highcut = cutoff / nyq
    b, a = scipy.signal.butter(order, highcut, btype='highpass', analog=False)
    return b, a

class HandReader:

    def __init__(self, independent=True):
        if independent:
            rospy.init_node('hand_reader', anonymous=True)  # Need unique name?
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, _romano_ctrl_init_helper)
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, lambda hand_state, digit_forces=digit_forces: _rcih(hand_state, digit_forces))
        self.digit_forces = None
        self.forces_lock = threading.Lock()
        self.init_finish = False
        self.hand_state = None
        self.running = True
        # In command line: 
        # rosrun topic_tools throttle messages /reflex_takktile/hand_state 50.0 /reflex_takktile/hand_state_throttle
        self.rate = rospy.Rate(50)  # Reflex hand has 50Hz sampling rate
        rospy.Subscriber("/reflex_takktile/hand_state", Hand, self._update_hand_state, queue_size=1)
        self.t_cb = rospy.Timer(rospy.Duration(1.0/50), self._callback_control)
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, self._callback_control, queue_size=1)
        # rospy.Subscriber("/reflex_takktile/hand_state_throttle", Hand, self._callback_control, queue_size=1)

        # rospy.Rate(0.25).sleep()  # allow 0.25 seconds of accumulation, is it possible rospy.Timer.sleep(0.25) sleeps all of ROS?
        time.sleep(0.25)

        print('Getting ReFlex TakkTile baseline for sensors...')

        # LOCK
        self.forces_lock.acquire()

        num_samples = self.digit_forces.shape[0]
        # num_samples = len(self.digit_forces)
        self.avg = np.sum([np.sum(self.digit_forces[i]) for i in range(num_samples)]) / (_NUM_SENSORS * 3 * num_samples)  # sensors per finger, 3 fingers

        # UNLOCK
        self.forces_lock.release()
        self.init_finish = True

        print('Done.')

    def _update_hand_state(self, hand_state):
        self.hand_state = hand_state

    def _callback_control(self, event):  # (self, hand_state)
        '''
        if self.init_finish:
            self._callback(hand_state)
        else:
            self._rcih(hand_state)
        self.rate.sleep()  # Try this instead of throttling
        '''
        if self.init_finish:
            self._callback(self.hand_state)
        else:
            self._rcih(self.hand_state)
        
    def _rcih(self, hand_state):
      # self.hand_state = hand_state
      # A priori TakkTile hand has 3 fingers
      new_forces = np.hstack((hand_state.finger[0].pressure, hand_state.finger[1].pressure, hand_state.finger[2].pressure))
      # LOCK
      self.forces_lock.acquire()
    
      if self.digit_forces is None:
          self.digit_forces = new_forces
      else:
          self.digit_forces = np.vstack((self.digit_forces, new_forces))
      # UNLOCK
      self.forces_lock.release()

    def collect_exp(self):
        b, a = scipy.signal.butter(1, _FILTER_CUTOFF/_NUM_SAMPLES, fs=_NUM_SAMPLES, btype='highpass')  # First (1) order Butterworth high pass (btype='highpass') filter with cutoff (5) Hz
        output = None
        loop_rate = rospy.Rate(50)  # Limit to 50hz loop rate
        for i in range(1200):
            # LOCK
            self.forces_lock.acquire()

            if self.digit_forces.shape[0] > _NUM_SAMPLES + _DIFF:
                self.digit_forces = self.digit_forces[-_NUM_SAMPLES - _DIFF:]  # Might be too slow to slice

            # UNLOCK
            self.forces_lock.release()

            next_reading = self.digit_forces[:_NUM_SAMPLES + _DIFF]
            if _DIFF:
                next_reading = np.diff(next_reading, axis=0)
            filtered = None
            if _FILTERING:
                filtered = scipy.signal.lfilter(b, a, next_reading)
		# Filter lines along which axis?
            else:
                filtered = next_reading
            if output is None:
                output = filtered[0]
            else:
                output = np.vstack((output, filtered[0]))
            loop_rate.sleep()

        diffstr = '_diff' if _DIFF else ''
        if _FILTERING:
            np.savetxt('{:d}hz_{:d}samples{:s}.out'.format(int(_FILTER_CUTOFF), _NUM_SAMPLES, diffstr), output)
        else:
            np.savetxt('unfiltered_samples{:s}.out'.format(diffstr), output)

    def collect_step(self, b, a, event):
        # LOCK
        self.forces_lock.acquire()

        if self.digit_forces.shape[0] > _NUM_SAMPLES + _DIFF:
            self.digit_forces = self.digit_forces[-_NUM_SAMPLES - _DIFF:]  # Might be too slow to slice

        # UNLOCK
        self.forces_lock.release()

        next_reading = self.digit_forces[:_NUM_SAMPLES + _DIFF]
        if _DIFF:
            next_reading = np.diff(next_reading, axis=0)
        filtered = None
        if _FILTERING:
            self.filtered = scipy.signal.lfilter(b, a, next_reading)
        else:
            self.filtered = next_reading

    def collect(self):
        b, a = scipy.signal.butter(1, _FILTER_CUTOFF/_NUM_SAMPLES, fs=_NUM_SAMPLES, btype='highpass')  # First (1) order Butterworth high pass (btype='highpass') filter with cutoff (5) Hz
        # loop_rate = rospy.Rate(50)  # Limit to 50hz loop rate, will this change anything on top of the callback control sleep?
        self.t = rospy.Timer(rospy.Duration(1.0/50), lambda e: self.collect_step(b, a, e))
        '''
        while self.running:
            # LOCK
            self.forces_lock.acquire()

            if self.digit_forces.shape[0] > _NUM_SAMPLES + _DIFF:
                self.digit_forces = self.digit_forces[-_NUM_SAMPLES - _DIFF:]  # Might be too slow to slice

            # UNLOCK
            self.forces_lock.release()

            next_reading = self.digit_forces[:_NUM_SAMPLES + _DIFF]
            if _DIFF:
                next_reading = np.diff(next_reading, axis=0)
            filtered = None
            if _FILTERING:
                self.filtered = scipy.signal.lfilter(b, a, next_reading)
            else:
                self.filtered = next_reading
            if output is None:
                output = self.filtered[-1]
            else:
                output = np.vstack((output, self.filtered[-1]))
            loop_rate.sleep()
        '''


    def _callback(self, hand_state):
        # self.hand_state = hand_state
        
        # new_reading = np.append(hand_state.finger[0].pressure, [hand_state.finger[1].pressure, hand_state.finger[2].pressure])
        new_reading = np.hstack((hand_state.finger[0].pressure, hand_state.finger[1].pressure, hand_state.finger[2].pressure))
        new_reading -= self.avg

        # LOCK
        self.forces_lock.acquire()

        self.digit_forces = np.vstack((self.digit_forces, new_reading))

        # UNLOCK
        self.forces_lock.release()


    def get_forces(self):
        last_read = self.digit_forces[-1]
        # return np.vstack(last_read[0:_NUM_SENSORS], last_read[_NUM_SENSORS:2*_NUM_SENSORS], last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])
        return np.sum(last_read[0:_NUM_SENSORS]), np.sum(last_read[_NUM_SENSORS:2*_NUM_SENSORS]), np.sum(last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])


    def get_forces_history(self):
        return np.sum(self.digit_forces, axis=1)


    def get_force_disturbances(self):
        last_read = self.filtered[-1]
        # return np.vstack(last_read[0:_NUM_SENSORS], last_read[_NUM_SENSORS:2*_NUM_SENSORS], last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])
        return np.sum(last_read[0:_NUM_SENSORS]), np.sum(last_read[_NUM_SENSORS:2*_NUM_SENSORS]), np.sum(last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])


    def stop():
        # self.running = False
        self.t.shutdown()
        self.t_cb.shutdown()


if __name__ == "__main__":
    # rospy.wait_for_message("/reflex_takktile/hand_state_throttle", Hand)
    reader = HandReader()
    reader.collect_exp()
