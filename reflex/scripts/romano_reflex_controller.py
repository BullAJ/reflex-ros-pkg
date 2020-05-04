#!/usr/bin/env python

import numpy as np
import scipy
import threading
import time
import rospy
from std_srvs.srv import Empty, Trigger, TriggerResponse
from std_msgs.msg import Bool
from reflex_msgs.msg import PoseCommand, VelocityCommand, ForceCommand, Hand, Finger
from bwhpf_ex import butter_bandpass_filter
from readings import HandReader, _NUM_FING, _NUM_SENSORS
from threading import Thread
from reflex_grasp import ReflexInterface


_MS_TO_RS = (
    np.pi / 180
)  # Need radius in this -- potentially use length of first finger link

# Useful Constants  # Original Units
_ATHRESH = 4.2  # m/(s^2)
_DLIMIT = 0.02 * 10  # N
_EFRICTION = 7.0  # N
_FBPTHRESH = 0.25  # N
_FLIMIT = 0.75  # N
_FTHRESH = 0.15  * 10 # N
_KD = 5000  # Ns/m
_KFCLOSE = 0.0013  # m/(Ns)
_KFOPEN = 0.0008  # m/(Ns)
_KHARDNESS = 0.027  * 10  # m/s
_KP = 20000  # N/m
_KSLIP = 1.08  # <None>
_SLIPTHRESH = 0.01  # <None>
_TSETTLE = 0.05  # s
_TUNLOAD = 0.20  # s
_VCLOSE = 0.04 * 30  # m/s
_VOPEN = 0.05 * 30  # m/s
_VTHRESH = 0.001 * 10  # m/s

_LOOP_HZ = 1000  # Defined by Romano, but we may change this

# TODO make all this into a class, and keep the following as instance variables:
# x_des, which I don't even know how to get -- maybe use distance when first contact is made, plus some error?
# Can use distal_approx from finger message, potentially
# TODO possibly reverse sign convention for velocity and distance? Fewer calculations would come of this.

# Possibly convert linear distances/velocities to angular (angles/angular velocities)
# Based on reflex_grasp, _VOPEN and _VCLOSE will likely need to be higher

# Documented changes on ReFlex TakkTile 2 vs 1:
# 14 sensors per finger vs 9
# TakkTile 2 has 8 (guess?) proximal, 6 (guess?) distal sensors per finger, TakkTile has 5 proximal, 4 distal
# v2 has IMU w/ it's own message embedded in each Finger. ( e.g. finger<i>.imu.quat for the quaternion (w, x, y, z) )
# Palm IMU: <HandMsg>.palmImu


class RomanoController:
    def __init__(self):
        rospy.init_node('romano_controller', anonymous=True)

        # Topics
        pos_cmd_topic = "/reflex_takktile/command_position"
        vel_cmd_topic = "/reflex_takktile/command_velocity"
        eff_cmd_topic = "/reflex_takktile/command_motor_force"

        # Services this class calls
        self.zero_tactile_srv = "/reflex_takktile/calibrate_tactile"
        self.calibrate_srv = "/reflex_takktile/calibrate_fingers"
        self.threshold_srv = "/reflex_takktile/set_tactile_threshold"
        self.enable_tactile_stops_srv = "/reflex_takktile/enable_tactile_stops"
        self.disable_tactile_stops_srv = "/reflex_takktile/disable_tactile_stops"

        self.rate = rospy.Rate(_LOOP_HZ)
        self.reader = HandReader(independent=False)
        self.v_des = -_VCLOSE  # TODO check negative convention

        # Set up our publishing to our chosen topics
        self.pos_cmd_pub = rospy.Publisher(pos_cmd_topic, PoseCommand, queue_size=1)
        self.vel_cmd_pub = rospy.Publisher(vel_cmd_topic, VelocityCommand, queue_size=1)
        self.eff_cmd_pub = rospy.Publisher(eff_cmd_topic, ForceCommand, queue_size=1)

    def get_motor_effort(self, x_g, x_des, v_g, v_des):
        # 'E' in Romano
        # Effort controller
        E = _KP * (x_g - x_des) + _KD * (v_g - v_des) - np.sign(v_des) * _EFRICTION
        return E

    def set_motor_effort(self, x_g, x_des, v_g, v_des=None):
        if v_des is None:
            v_des = self.v_des
        E = self.get_motor_effort(x_g, x_des, v_g, v_des)
        """
        TODO hand_state.motor[i].load gives the torque (?) on the motor.
        In some cases we calculate motor effort off of pressure readings, and we
        may have to do some slick conversion to get from N*m to N on pressure.
        Also, TODO compensate for the fact that TakkTile sensor readings are unitless
        """
        eff_cmd = ForceCommand()
        eff_cmd.f1 = E
        eff_cmd.f2 = E
        eff_cmd.f3 = E
        eff_cmd.preshape = 0.0
        self.eff_cmd_pub.publish(eff_cmd)

    def set_motor_effort_direct(self, E):
        eff_cmd = ForceCommand()
        eff_cmd.f1 = E
        eff_cmd.f2 = E
        eff_cmd.f3 = E
        eff_cmd.preshape = 0.0
        self.eff_cmd_pub.publish(eff_cmd)

    def get_v_des(self, F_min, F_des=None):
        if F_des is None:
            F_des = self.F_des
        KF = _KFCLOSE if F_min - F_des < 0 else _KFOPEN
        return KF * (F_min - F_des)

    def vel_pub(self, event=None, v=None):
        """
        While the internal representation holds to Romano's conventions of
        negative velocity => closing, the reflex hand has the opposite convention,
        so this function exists to abstract that detail away from the rest of
        the functions. This way, we can remain as close as possible to Romano's
        representation.
        """
        if v is None:
            v = self.v_des
        vel_cmd = VelocityCommand()
        # Potentially halve each velocity, since the closure is coming from both directions?
        vel_cmd.f1 = -v
        vel_cmd.f2 = -v
        vel_cmd.f3 = -v
        vel_cmd.preshape = 0.0
        self.vel_cmd_pub.publish(vel_cmd)

    def close(self):
        """
        A complex model of what Romano dictates to be contact
        """
        print("In Close stage")
        F_d_1, F_d_2, F_s = self.reader.get_forces()
        F_d = (F_d_1 + F_d_2) / 2
        F_d_dist, F_s_dist = self.reader.get_force_disturbances()
        # Get accelerations from KUKA arm wrench?

        # We can use this more complicated setup, or simply use the Reflex tactile stops
        d_contact = F_d > _FLIMIT or F_d_dist > _DLIMIT
        s_contact = F_s > _FLIMIT or F_s_dist > _DLIMIT
        vel_cmd = VelocityCommand()
        while not (d_contact and s_contact):
            F_d_1, F_d_2, F_s = self.reader.get_forces()
            F_d = (F_d_1 + F_d_2) / 2
            F_d_dist, F_s_dist = self.reader.get_force_disturbances()
            # Accelerations
            d_contact = F_d > _FLIMIT or F_d_dist > _DLIMIT
            s_contact = F_s > _FLIMIT or F_s_dist > _DLIMIT
            # Move with desired velocity
            self.vel_pub()

            self.rate.sleep()

    def close_simple(self):
        """
        Pre-existing model uses force thresholds as well to determine contacts, but not 
        disturbance limits
        """
        print("In Close stage")
        # self.interface._close_hand()
        try:
            zero_tactile = rospy.ServiceProxy(self.zero_tactile_srv, Empty)
            zero_tactile()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to zero ReFlex tactile failed: %s" % e)
        '''
        try:
            pressure = [[_FLIMIT] * _NUM_SENSORS] * _NUM_FING
            set_thresh = rospy.ServiceProxy(self.threshold_srv, Empty)  # TODO Figure out if Empty needs to be something else?
            set_thresh(pressure)
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to set tactile threshold failed: %s" % e)
        '''
        try:
            enable_stops = rospy.ServiceProxy(self.enable_tactile_stops_srv, Empty)
            enable_stops()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to enable tactile stops failed: %s" % e)

        for i in range(1000):  # Is the loop necessary?
            self.vel_pub()
            self.rate.sleep()

        try:
            disable_stops = rospy.ServiceProxy(self.disable_tactile_stops_srv, Empty)
            disable_stops()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to disable tactile stops failed: %s" % e)

    def load(self):
        print("In Load stage")
        time.sleep(_TSETTLE)

        F_c = max(self.reader.get_forces_history()) * (
            _KHARDNESS / _VCLOSE
        )  # F_des can be determined a priori
        self.F_des = F_c

        F_min = min(self.reader.get_forces())

        # Velocities
        # Max or avg?
        '''
        v_g_d = max(
            self.reader.hand_state.motor[0].velocity,
            self.reader.hand_state.motor[1].velocity,
        )
        '''
        v_g_d = 0.5 * (self.reader.hand_state.motor[0].velocity + self.reader.hand_state.motor[1].velocity)
        v_g_s = self.reader.hand_state.motor[2].velocity
        # Reflex convetion is positive => closing, but we want positive velocity to mean opening.
        v_g = -(v_g_d + v_g_s)

        stable_contact = (abs(F_c - F_min) < _FTHRESH) and (abs(v_g) < _VTHRESH)
        while not stable_contact:
            # Switch to force/position control
            # TODO remove print
            print(_FTHRESH - abs(F_c - F_min), _VTHRESH - abs(v_g))
            '''
            x_g_d = max(
                self.reader.hand_state.finger[0].proximal,
                self.reader.hand_state.finger[1].proximal,
            )
            '''
            x_g_d = 0.5 * (self.reader.hand_state.finger[0].proximal + self.reader.hand_state.finger[1].proximal)
            x_g_s = self.reader.hand_state.finger[2].proximal
            # If following convention, increases in these x will correspond to more flexion
            x_g = (
                (2 * np.pi) - x_g_d - x_g_s
            )  # 2pi means gripper is fully open, 0 closed
            # TODO remove print
            print("Setting motor effort")
            self.set_motor_effort(
                x_g, 0, v_g, v_des=self.get_v_des(F_min)
            )  # F_des is set to F_c, have no idea how to get x_des

            F_min = min(self.reader.get_forces())
            # Max or avg?
            '''
            v_g_d = max(
                self.reader.hand_state.motor[0].velocity,
                self.reader.hand_state.motor[1].velocity,
            )
            '''
            v_g_d = 0.5 * (self.reader.hand_state.motor[0].velocity + self.reader.hand_state.motor[1].velocity)
            v_g_s = self.reader.hand_state.motor[2].velocity
            v_g = -(v_g_d + v_g_s)
            stable_contact = (abs(F_c - F_min) < _FTHRESH) and (abs(v_g) < _VTHRESH)

            self.rate.sleep()

    def lift_hold(self):
        # Need this to happen potentially multiple times while holding and moving
        F_d_1, F_d_2, F_s = self.reader.get_forces()
        F_d = (F_d_1 + F_d_2) / 2
        F = (F_d + F_s) / 2
        F_d_dist, F_s_dist = self.reader.get_force_disturbances()
        F_dist = (F_d_dist + F_s_dist) / 2

        # First-order Chebyshev bandpass filter, 1-5Hz
        F_hist = self.reader.get_forces_history() / 2
        nyq = 0.5 * len(F_hist)
        b, a = scipy.signal.cheby2(1, [1 / nyq, 5 / nyq], btype="bandpass")
        FBP = scipy.signal.lfilter(b, a, F_hist)[-1]

        slip = (abs(F_dist) > F * _SLIPTHRESH) and (FBP < _FBPTHRESH)
        if slip:
            # F_c *= _KSLIP
            self.F_des *= (
                _KSLIP  # Current F_des is F_c, carried over from previous step.
            )

    def lift_hold_loop(self):
        """
        A dummy example of lifting and holding for some amount of time. This will become defunct
        once a real transition method is added.
        """
        print("In Lift/Hold stage")
        s = 3  # minimum/approximate loop duration in seconds
        for i in range(s * 10):
            self.lift_hold()
            self.rate.sleep()

    def replace(self):
        # Still account for slips as in previous phase?
        print("In Replace stage")  # Acceleration-based by Romano

    def unload(self):
        print("In Unload stage")
        begin = rospy.get_rostime()
        F_c = max(self.reader.get_forces_history()) * (
            _KHARDNESS / _VCLOSE
        )  # F_des can be determined a priori
        while (rospy.get_rostime() - begin) < _TUNLOAD:
            self.F_des = F_c * (1 - (rospy.get_rostime() - begin) / _TUNLOAD)
            # After computing new desired force, hand new desired force to controller
            self.set_motor_effort_direct(self.F_des)

    def open_hand(self):
        print("In Open stage")
        self.v_des = _VOPEN
        for i in range(10):
            # self.vel_cmd_pub.publish(vel_cmd)
            self.vel_pub()

    def main(self):
        # TODO do we need explicit threads with the rospy.Timer going on?
        '''
        reader_thread = Thread(target=self.reader.collect)
        reader_thread.start()
        # Cleanup
        rospy.on_shutdown(self.reader.stop)
        rospy.on_shutdown(reader_thread.join)
        '''
        self.reader.collect()
        rospy.on_shutdown(self.reader.stop)

        # loop_rate = self.rate  # 1kHz control loop
        functions = [self.close_simple, self.load, self.lift_hold_loop, self.replace, self.unload, self.open_hand]
        # TODO instead of a loop, find a way to issue commands to the controller
        # TODO possibly have position/force controllers running in a different thread? Each in their own?
        while not rospy.is_shutdown():
            for i in range(len(functions)):
                functions[i]()
                # loop_rate.sleep()


if __name__ == "__main__":
    controller = RomanoController()
    controller.main()
