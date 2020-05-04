#!/usr/bin/env python

import numpy as np
import scipy
import threading
import time
import rospy
from std_srvs.srv import Empty, Trigger, TriggerResponse
from std_msgs.msg import Bool
from bwhpf_ex import butter_bandpass_filter

_HAND_VERSION = ""
from reflex_msgs2.msg import PoseCommand, VelocityCommand, ForceCommand, Hand, Finger
_NUM_SENSORS = 14  # per finger
_USE_ACCEL = True
if _HAND_VERSION != "2":
    from reflex_msgs.msg import PoseCommand, VelocityCommand, ForceCommand, Hand, Finger
    _NUM_SENSORS = 9  # per finger
    _USE_ACCEL = False

_FILTERING = True
_DIFF = 0
_NUM_SAMPLES = 100  # The reflex hand_state topic gives us 20hz - i.e. 50 gives 2.5 seconds of history)
_HIGHCUT = 5.0  # Hz
_LOWCUT = 5.0  # Hz, best found 5.0
_HIGHCHEBY = 5.0  # Hz
_LOWCHEBY = 1.0  # Hz
_ACC_HIGHCUT = 15.0  # Hz, best found 15.0
_RIPPLE  = 20  # Maximum allowable ripple for chebyshev filter

_LOOP_HZ = 20

_NUM_FING = 3
_NUM_DISTALS = _NUM_SENSORS - 5  # distal sensors per finger, true for either version

def butter_highpass(cutoff=_HIGHCUT, fs=_NUM_SAMPLES, order=1):
    nyq = 0.5 * fs
    highcut = cutoff / nyq
    b, a = scipy.signal.butter(order, highcut, btype='highpass', analog=False)
    return b, a

def butter_lowpass(cutoff=_LOWCUT, fs=_NUM_SAMPLES, order=1):
    nyq = 0.5 * fs
    lowcut = cutoff / nyq
    b, a = scipy.signal.butter(order, lowcut, btype='lowpass', analog=False)
    return b, a

def cheby1_bandpass(lo=_LOWCHEBY, hi=_HIGHCHEBY, ripple=_RIPPLE, fs=_NUM_SAMPLES + _DIFF):
    nyq = 0.5 * fs
    cheby_b, cheby_a = scipy.signal.cheby1(1, ripple, [lo / nyq, hi / nyq], btype="bandpass")  # cheby2 gives inverse cheby filter
    return cheby_b, cheby_a

class HandReader:

    def __init__(self, independent=True):
        if independent:
            rospy.init_node('hand_reader', anonymous=True)  # Need unique name?
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, _romano_ctrl_init_helper)
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, lambda hand_state, digit_forces=digit_forces: _rcih(hand_state, digit_forces))
        self.digit_forces = None
        self.lofilt = None
        self.hifilt = None
        self.bpfilt = None
        self.acc = None
        self.hifilt_acc = None
        self.forces_lock = threading.Lock()

        self.acc_lock = threading.Lock()

        self.init_finish = False
        self.hand_state = None
        # self.running = True
        # In command line: 
        # rosrun topic_tools throttle messages /reflex_takktile/hand_state 50.0 /reflex_takktile/hand_state_throttle

        # Build filters for sensors
        self.lob, self.loa = butter_lowpass()
        self.hib, self.hia = butter_highpass()
        self.bpb, self.bpa = cheby1_bandpass()

        # Build filter for acceleration
        self.acc_b, self.acc_a = butter_highpass(cutoff=_ACC_HIGHCUT)

        # self.rate = rospy.Rate(_LOOP_HZ)  # Reflex hand has 50Hz sampling rate
        # TODO check buff_size in the following line, see if it actually reduces delay based on size
        rospy.Subscriber("/reflex_takktile%s/hand_state" % _HAND_VERSION, Hand, self._update_hand_state, queue_size=1)
        self.t_cb = rospy.Timer(rospy.Duration(1.0/_LOOP_HZ), self._callback_control)
        # rospy.Subscriber("/reflex_takktile/hand_state", Hand, self._callback_control, queue_size=1)
        # rospy.Subscriber("/reflex_takktile/hand_state_throttle", Hand, self._callback_control, queue_size=1)

        rospy.wait_for_message("/reflex_takktile%s/hand_state" % _HAND_VERSION, Hand)
        rospy.loginfo('Getting baseline for ReFlex TakkTile sensors...')
        # rospy.Rate(0.25).sleep()  # allow 0.25 seconds of accumulation, is it possible rospy.Timer.sleep(0.25) sleeps all of ROS?
        time.sleep(0.25)

        # LOCK
        self.forces_lock.acquire()

        num_samples = self.digit_forces.shape[0]
        # num_samples = len(self.digit_forces)
        self.avg = np.sum([np.sum(self.digit_forces[i]) for i in range(num_samples)]) / (_NUM_SENSORS * 3 * num_samples)  # sensors per finger, 3 fingers

        # UNLOCK
        self.forces_lock.release()
        self.init_finish = True

        rospy.loginfo('Done.')

    def _update_hand_state(self, hand_state):
        self.hand_state = hand_state
        # Possibly put one-shot timer in here, instead of calling _callback_control with a separate timer?

    def _callback_control(self, event):  # (self, hand_state)
        '''
        if self.init_finish:
            self._callback(hand_state)
        else:
            self._rcih(hand_state)
        self.rate.sleep()  # Try this instead of throttling
        '''
        if self.hand_state is None:
            return
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

        if _USE_ACCEL:
            # LOCK ACC
            self.acc_lock.acquire()
            new_accelerations = hand_state.palmImu.quat[:3]
            if self.acc is None:
                self.acc = new_accelerations
            else:
                self.acc = np.vstack((self.acc, new_accelerations))
                if self.acc.shape[0] > _NUM_SAMPLES:
                    self.acc = self.acc[-_NUM_SAMPLES:]
            self.hifilt_acc = self.acc
            # UNLOCK ACC
            self.acc_lock.release()

    def collect_exp_step(self, event):
        # LOCK
        # self.forces_lock.acquire()

        # if self.digit_forces.shape[0] > _NUM_SAMPLES + _DIFF:
        #     self.digit_forces = self.digit_forces[-_NUM_SAMPLES - _DIFF:]  # Might be too slow to slice

        # UNLOCK
        # self.forces_lock.release()

        next_reading = self.digit_forces[:_NUM_SAMPLES + _DIFF]
        if _DIFF:
            next_reading = np.diff(next_reading, axis=0)
        if _FILTERING:
                # TODO possibly use filtfilt to get 0 phase?
            if self.lofilt is None:
                self.lofilt = scipy.signal.lfilter(self.lob, self.loa, next_reading, axis=0)[-1]
            else:
                self.lofilt = np.vstack((self.lofilt,scipy.signal.lfilter(self.lob, self.loa, next_reading, axis=0)[-1]))
            if self.hifilt is None:
                self.hifilt = scipy.signal.lfilter(self.hib, self.hia, next_reading, axis=0)[-1]
            else:
                self.hifilt = np.vstack((self.bpfilt,scipy.signal.lfilter(self.hib, self.hia, next_reading, axis=0)[-1]))
            if self.bpfilt is None:
                self.bpfilt = scipy.signal.lfilter(self.bpb, self.bpb, next_reading, axis=0)
            else:
                self.bpfilt = np.vstack((self.bpfilt,scipy.signal.lfilter(self.bpb, self.bpb, next_reading, axis=0)[-1]))
        else:
            self.lofilt = next_reading
            self.hifilt = next_reading
            self.bpfilt = next_reading

        if _USE_ACCEL:
            # LOCK ACC
            # self.acc_lock.acquire()

            # if self.acc is not None and self.acc.shape[0] > _NUM_SAMPLES:
            #     self.acc = self.acc[-_NUM_SAMPLES:]

            # UNLOCK ACC
            # self.acc_lock.release()

            next_reading = self.acc[:_NUM_SAMPLES]
            if _FILTERING:
                if self.hifilt_acc is None:
                    self.hifilt_acc = scipy.signal.lfilter(self.acc_b, self.acc_a, next_reading, axis=0)[-1]
                else:
                    self.hifilt_acc = np.vstack((self.hifilt_acc,scipy.signal.lfilter(self.acc_b, self.acc_a, next_reading, axis=0)[-1]))
            else:
                self.hifilt_acc = next_reading

    def collect_exp(self):
        rospy.Timer(rospy.Duration(1.0/_LOOP_HZ), self.collect_exp_step)

    def dump_structures_to_file(self):
        if _USE_ACCEL:
            np.savetxt('acc.csv', self.acc, delimiter=",")
            np.savetxt('hifilt_acc_%d.csv' % int(_ACC_HIGHCUT), self.hifilt_acc, delimiter=",")
        np.savetxt('digit_forces.csv', self.digit_forces, delimiter=",")
        np.savetxt('lofilt_f_%d.csv' % int(_LOWCUT), self.lofilt, delimiter=",")
        np.savetxt('hifilt_f_%d.csv' % int(_HIGHCUT), self.hifilt, delimiter=",")
        np.savetxt('bpfilt_f_%d_%d.csv' % (int(_LOWCHEBY), int(_HIGHCHEBY)), self.bpfilt, delimiter=",")

    def collect_step(self, event):
        # LOCK
        self.forces_lock.acquire()

        if self.digit_forces.shape[0] > _NUM_SAMPLES + _DIFF:
            self.digit_forces = self.digit_forces[-_NUM_SAMPLES - _DIFF:]  # Might be too slow to slice

        # UNLOCK
        self.forces_lock.release()

        next_reading = self.digit_forces[:_NUM_SAMPLES + _DIFF]
        if _DIFF:
            next_reading = np.diff(next_reading, axis=0)
        if _FILTERING:
            # TODO possibly use filtfilt to get 0 phase?
            self.lofilt = scipy.signal.lfilter(self.lob, self.loa, next_reading, axis=0)
            self.hifilt = scipy.signal.lfilter(self.hib, self.hia, next_reading, axis=0)
            self.bpfilt = scipy.signal.lfilter(self.bpb, self.bpb, next_reading, axis=0)
        else:
            self.lofilt = next_reading
            self.hifilt = next_reading
            self.bpfilt = next_reading

        # LOCK ACC
        self.acc_lock.acquire()

        if self.acc.shape[0] > _NUM_SAMPLES:
            self.acc = self.acc[-_NUM_SAMPLES:]

        # UNLOCK ACC
        self.acc_lock.release()

        next_reading = self.acc[:_NUM_SAMPLES]
        if _FILTERING:
            self.hifilt_acc = scipy.signal.lfilter(self.acc_b, self.acc_a, next_reading, axis=0)
        else:
            self.hifilt_acc = next_reading

    def collect(self):
        # loop_rate = rospy.Rate(_LOOP_HZ)  # Limit to 50hz loop rate, will this change anything on top of the callback control sleep?
        self.t = rospy.Timer(rospy.Duration(1.0/_LOOP_HZ), self.collect_step)

        # IMU samples at a rate of 100Hz (?)
        # For now, use same filter as before
        # self.imu_t = rospy.Timer(rospy.duration(1.0/100), lambda e: self.collect_accelerations(b, a, e))
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
                self.hifilt = scipy.signal.lfilter(b, a, next_reading, axis=0)
            else:
                self.hifilt = next_reading
            if output is None:
                output = self.hifilt[-1]
            else:
                output = np.vstack((output, self.hifilt[-1]))
            loop_rate.sleep()
        '''


    def _callback(self, hand_state):
        # self.hand_state = hand_state
        
        # new_reading = np.append(hand_state.finger[0].pressure, [hand_state.finger[1].pressure, hand_state.finger[2].pressure])
        new_reading = np.hstack((hand_state.finger[0].pressure, hand_state.finger[1].pressure, hand_state.finger[2].pressure))
        # new_reading -= self.avg  # Necessary? Given calibrations, this may not need to be done.

        # LOCK
        self.forces_lock.acquire()

        self.digit_forces = np.vstack((self.digit_forces, new_reading))

        # UNLOCK
        self.forces_lock.release()

        if _USE_ACCEL:
            new_acc = hand_state.palmImu.quat[:3]
            # LOCK ACC
            self.acc_lock.acquire()
            
            self.acc = np.vstack((self.acc, new_acc))

            # UNLOCK ACC
            self.acc_lock.release()


    def get_forces(self, last_read=None):
        '''
        Gets the maximum force along a finger from the latest reading - Use this for immediate control
        '''
        if last_read is None:
            last_read = self.digit_forces[-1]
        # return np.vstack(last_read[0:_NUM_SENSORS], last_read[_NUM_SENSORS:2*_NUM_SENSORS], last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])
        # return np.sum(last_read[0:_NUM_SENSORS]), np.sum(last_read[_NUM_SENSORS:2*_NUM_SENSORS]), np.sum(last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])
        ret = (np.max(last_read[0:_NUM_SENSORS]), np.max(last_read[_NUM_SENSORS:2*_NUM_SENSORS]), np.max(last_read[2*_NUM_SENSORS:3*_NUM_SENSORS]))
        return np.vstack(ret)

    def get_forces_lofilt(self):
        '''
        Gets the maximum force along a finger from the latest reading - Use this for immediate control
        '''
        return self.get_forces(last_read=self.lofilt[-1])

    def get_forces_hifilt(self):
        '''
        Gets the maximum force along a finger from the latest reading - Use this for immediate control
        '''
        return self.get_forces(last_read=self.hifilt[-1])

    def get_forces_bpfilt(self):
        '''
        Gets the maximum force along a finger from the latest reading - Use this for immediate control
        '''
        return self.get_forces(last_read=self.bpfilt[-1])

    def get_forces_sum(self):
        include = 8
        last_read = self.digit_forces[-1]
        prox1sum = np.sum(last_read[:include])
        prox2sum = np.sum(last_read[_NUM_SENSORS:_NUM_SENSORS + include])
        prox3sum = np.sum(last_read[2*_NUM_SENSORS:2*_NUM_SENSORS + include])
        return np.vstack((prox1sum, prox2sum, prox3sum))

    def get_forces_max(self):
        '''
        Gets the max over the entire present history of forces - Use this to establish a baseline
        '''
        # Include x number of sensors for a finger.
        # On both hands, the first 8 (5 prox/3 dist)
        # are the most forward-facing sensors
        include = 8
        prox1max = np.max(self.digit_forces[:,:include])
        prox2max = np.max(self.digit_forces[:,_NUM_SENSORS:_NUM_SENSORS + include])
        prox3max = np.max(self.digit_forces[:,2*_NUM_SENSORS:2*_NUM_SENSORS + include])
        return np.vstack((prox1max, prox2max, prox3max))

    def get_forces_max_lofilt(self):
        '''
        Gets the max over the entire present history of forces - Use this to establish a baseline
        '''
        # Include x number of sensors for a finger.
        # On both hands, the first 8 (5 prox/3 dist)
        # are the most forward-facing sensors
        include = 8
        prox1max = np.max(self.lofilt[:,:include])
        prox2max = np.max(self.lofilt[:,_NUM_SENSORS:_NUM_SENSORS + include])
        prox3max = np.max(self.lofilt[:,2*_NUM_SENSORS:2*_NUM_SENSORS + include])
        return np.vstack((prox1max, prox2max, prox3max))

    def get_forces_max_sum(self):
        '''
        Returns the maximum sum of sensor readings for each finger
        '''
        include = 8
        prox1sum = np.max(np.sum(self.digit_forces[:,:include], axis=1))
        prox2sum = np.max(np.sum(self.digit_forces[:,_NUM_SENSORS:_NUM_SENSORS + include], axis=1))
        prox3sum = np.max(np.sum(self.digit_forces[:,2*_NUM_SENSORS:2*_NUM_SENSORS + include], axis=1))
        return np.vstack((prox1sum, prox2sum, prox3sum))

    def get_forces_history(self):
        return np.sum(self.digit_forces, axis=1)

    def get_force_disturbances(self):
        last_read = self.hifilt[-1]
        # return np.vstack(last_read[0:_NUM_SENSORS], last_read[_NUM_SENSORS:2*_NUM_SENSORS], last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])
        return np.sum(last_read[0:_NUM_SENSORS]), np.sum(last_read[_NUM_SENSORS:2*_NUM_SENSORS]), np.sum(last_read[2*_NUM_SENSORS:3*_NUM_SENSORS])

    def get_acc(self):
        return self.acc[-1]

    def get_acc_norm(self):
        return np.linalg.norm(self.acc[-1])
    
    def get_acc_norm_hist(self):
        return np.linalg.norm(self.acc, axis=1)

    def get_acc_hifilt(self):
        return self.hifilt_acc[-1]

    def get_acc_norm_hifilt(self):
        return np.linalg.norm(self.hifilt_acc[-1])

    def get_acc_norm_hifilt_hist(self):
        return np.linalg.norm(self.hifilt_acc, axis=1)[-1]

    def get_acc_norm_then_filter(self):
        norm = np.linalg.norm(self.acc, axis=1)
        return scipy.signal.lfilter(self.acc_b, self.acc_a, norm)[-1]

    def get_acc_shapes(self):
        shape1 = len(self.acc) if type(self.acc) is tuple else self.acc.shape
        shape2 = len(self.hifilt_acc) if type(self.hifilt_acc) is tuple else self.hifilt_acc.shape
        return shape1, shape2

    def get_finger_poses(self):
        return np.vstack([self.hand_state.finger[i].proximal for i in range(3)])

    def get_finger_poses_motor(self):
        return np.vstack([self.hand_state.motor[i].joint_angle for i in range(3)])

    def get_motor_angles(self):
        return np.vstack([self.hand_state.motor[i].raw_angle for i in range(3)])

    def all_contact(self):
        f1_in_contact = sum(self.hand_state.finger[0].contact) > 0
        f2_in_contact = sum(self.hand_state.finger[1].contact) > 0
        f3_in_contact = sum(self.hand_state.finger[2].contact) > 0
        return f1_in_contact and f2_in_contact and f3_in_contact

    def all_fingers_stopped(self, stop_threshold=10.0):
        f1_stopped = self.hand_state.motor[0].velocity < stop_threshold
        f2_stopped = self.hand_state.motor[1].velocity < stop_threshold
        f3_stopped = self.hand_state.motor[2].velocity < stop_threshold
        return f1_stopped and f2_stopped and f3_stopped

    def stop(self):
        self.running = False
        self.t.shutdown()
        self.t_cb.shutdown()

if __name__ == "__main__":
    # rospy.wait_for_message("/reflex_takktile/hand_state_throttle", Hand)
    reader = HandReader()
    # rospy.on_shutdown(reader.dump_structures_to_file)
    reader.collect_exp()
    rospy.spin()
    reader.dump_structures_to_file()