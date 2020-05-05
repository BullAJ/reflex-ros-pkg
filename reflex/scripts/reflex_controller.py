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


_MS_TO_RS = (
    np.pi / 180
)  # Need radius in this -- potentially use length of first finger link

# Useful Constants  # Original Units
_ATHRESH = 4.2  # m/(s^2)
_DLIMIT = 0.02 * 10  # N
_EFRICTION = 7.0  # N
_FBPTHRESH = 0.25  # N
_FLIMIT = 0.75  # N
_FTHRESH = 0.15 * 10  # N
_KD = 5000  # Ns/m
_KFCLOSE = 0.0013  # m/(Ns)
_KFOPEN = 0.0008  # m/(Ns)
_KHARDNESS = 0.027 * 10  # m/s
_KP = 20000  # N/m
_KSLIP = 1.08  # <None>
_SLIPTHRESH = 0.01 * 100  # <None>
_TSETTLE = 0.05  # s
_TUNLOAD = 0.20  # s
_VCLOSE = 0.04 * 30  # m/s
_VOPEN = 0.05 * 30  # m/s
_VTHRESH = 0.001 * 10  # m/s

# Constants made for LL4MA
_TSETTLE = 0.1  # Changed from Romano
_KF = 0.0036  # The gain on the control signal, 0.002 is too low for firm objects
_FERR = 0.04  #* _F_GAINS  # The minimum amount we adjust any given finger in control, ends up being useless with window
_LOWINDOW = -5  # The lower bound on deltas we allow for forces read from the fingers vs desires
_HIWINDOW = 40  # The upper bound ''
_KSLIP *= 1  # Same _KSLIP as Romano, for now
_ATHRESH = 0.008  # m/(s^2)

_LOOP_HZ = 20  # Defined by Romano as 1kHz, but we may change this

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


class ReflexController:
    def __init__(self):
        rospy.init_node("reflex_controller", anonymous=True)

        prefix = "/reflex_takktile%s/" % _HAND_VERSION

        # Topics
        pos_cmd_topic = prefix + "command_position"
        vel_cmd_topic = prefix + "command_velocity"
        eff_cmd_topic = prefix + "command_motor_force"

        # Services this class calls
        self.zero_tactile_srv = prefix + "calibrate_tactile"
        self.calibrate_srv = prefix + "calibrate_fingers"
        self.threshold_srv = prefix + "set_tactile_threshold"
        self.enable_tactile_stops_srv = prefix + "enable_tactile_stops"
        self.disable_tactile_stops_srv = prefix + "disable_tactile_stops"

        # Set up our publishing to our chosen topics
        self.pos_cmd_pub = rospy.Publisher(pos_cmd_topic, PoseCommand, queue_size=1)
        self.vel_cmd_pub = rospy.Publisher(vel_cmd_topic, VelocityCommand, queue_size=1)
        self.eff_cmd_pub = rospy.Publisher(eff_cmd_topic, ForceCommand, queue_size=1)

        # Set hand to zero pose and calibrate
        self.control_direct([0, 0, 0])
        try:
            zero_tactile = rospy.ServiceProxy(self.zero_tactile_srv, Empty)
            zero_tactile()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to zero ReFlex tactile failed: %s" % e)

        try:
            calibrate_fingers = rospy.ServiceProxy(self.calibrate_srv, Empty)
            calibrate_fingers()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to calibrate ReFlex tactile failed: %s" % e)

        rospy.sleep(5)  # Give time for calibration to complete

        self.rate = rospy.Rate(_LOOP_HZ)
        self.reader = HandReader(independent=False)
        self.v_des = _VCLOSE
        self.athresh = _ATHRESH
        self.replace_calls = 0


    def adjust_pose(self, F_observed, F_des):
        # 'E' in Romano
        # Control law
        # Hack: Increase desired F for finger 3, and/or give it more error
        delta = _KF * (F_des - F_observed)
        return np.where(abs(delta) < _FERR, 0, delta)  # 0 the delta if it's within tolerance

    def control1(self, F_observed, F_des=None):
        # Refresh pose from state
        next_pose_pre = self.reader.get_finger_poses()
        if F_des is None:
            F_des = self.F_des
        delta_pose = self.adjust_pose(F_observed, F_des)
        next_pose = next_pose_pre + delta_pose

        # If delta for a finger is zero, maintain that pose internally
        self.pose = np.where(delta_pose == 0, self.pose, next_pose)
        # self.pose = np.where(next_pose - self.pose < delta_pose, self.pose, next_pose)  # Does this actually help with drift?

        pos_cmd = PoseCommand()
        pos_cmd.f1 = next_pose[0]  # self.pose[0]
        pos_cmd.f2 = next_pose[1]  # self.pose[1]
        pos_cmd.f3 = next_pose[2]  # self.pose[2]
        pos_cmd.preshape = 0.0
        self.pos_cmd_pub.publish(pos_cmd)

    def control2(self, F_observed, F_des=None):
        if F_des is None:
            F_des = self.F_des
        current_pose = self.reader.get_finger_poses_motor()
        # Calculate delta
        delta = F_observed - self.F_des
        for i in range(3):
            '''
            if delta[i] < _LOWINDOW:
                self.pose[i] = current_pose[i] - _KF * (delta[i] - _LOWINDOW)
                self.finger_adjustment_states[i] = 1  # Tightening or loosening the joint
            elif delta[i] > _HIWINDOW:
                self.pose[i] = current_pose[i] - _KF * (delta[i] - _HIWINDOW)
                self.finger_adjustment_states[i] = 1
            '''
            if delta[i] < _LOWINDOW or delta[i] > _HIWINDOW:
                self.pose[i] = current_pose[i] - _KF * delta[i]
                self.finger_adjustment_states[i] = 1
            elif self.finger_adjustment_states[i] != 0:  # We changed state to a stable condition - force reading is within window and we were previously adjusting
                self.pose[i] = current_pose[i]
                self.finger_adjustment_states[i] = 0

        pos_cmd = PoseCommand()
        pos_cmd.f1 = self.pose[0]
        pos_cmd.f2 = self.pose[1]
        pos_cmd.f3 = self.pose[2]
        pos_cmd.preshape = 0.0
        self.pos_cmd_pub.publish(pos_cmd)

        return sum(self.finger_adjustment_states) == 0  # All fingers are not being adjusted => all fingers stable


    def control_direct(self, pose):
        pos_cmd = PoseCommand()
        pos_cmd.f1 = pose[0]
        pos_cmd.f2 = pose[1]
        pos_cmd.f3 = pose[2]
        pos_cmd.preshape = 0.0
        self.pos_cmd_pub.publish(pos_cmd)

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
        vel_cmd.f1 = v
        vel_cmd.f2 = v
        vel_cmd.f3 = v
        vel_cmd.preshape = 0.0
        self.vel_cmd_pub.publish(vel_cmd)

    def close(self):
        """
        A complex model of what Romano dictates to be contact
        """
        print("In Close stage")
        # TODO make more complex model (perhaps?)
        # Explicit contact stop?
        self.close_simple()

    def close_simple(self):
        """
        Pre-existing model uses force thresholds as well to determine contacts, but not 
        disturbance limits
        """
        rospy.loginfo("In Close stage")
        # self.interface._close_hand()
        """
        try:
            pressure = [[_FLIMIT] * _NUM_SENSORS] * _NUM_FING
            set_thresh = rospy.ServiceProxy(self.threshold_srv, Empty)  # TODO Figure out if Empty needs to be something else?
            set_thresh(pressure)
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to set tactile threshold failed: %s" % e)
        """
        try:
            enable_stops = rospy.ServiceProxy(self.enable_tactile_stops_srv, Empty)
            enable_stops()
            rospy.loginfo("Enabled Tactile Stops")
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to enable tactile stops failed: %s" % e)

        # Revise this approach. Try to find out when fingers make contact.
        self.vel_pub()
        # Wait until all fingers are moving
        while self.reader.all_fingers_stopped() and not rospy.is_shutdown():
            self.rate.sleep()
        # Wait until they are all stopped
        while not self.reader.all_fingers_stopped() and not rospy.is_shutdown():
            self.rate.sleep()
        # while not self.reader.all_contact() and not rospy.is_shutdown():
            # self.rate.sleep()

        try:
            disable_stops = rospy.ServiceProxy(self.disable_tactile_stops_srv, Empty)
            disable_stops()
            rospy.loginfo("Disabled Tactile Stops")
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to disable tactile stops failed: %s" % e)

        self.finger_adjustment_states = [1, 1, 1]  # Assume all fingers will need to be adjusted to start with
        self.pose = self.reader.get_finger_poses_motor()

    def load_simple(self):
        '''
        Simpler Load phase based on reflex_grasp2.py
        '''
        rospy.loginfo("In Load stage (simple version)")
        rospy.sleep(_TSETTLE)

        angles = self.reader.get_finger_poses()
        pos_increment = 0.8
        pos_cmd = PoseCommand()
        pos_cmd.f1 = angles[0] + pos_increment
        pos_cmd.f2 = angles[1] + pos_increment
        pos_cmd.f3 = angles[2] + pos_increment
        pos_cmd.preshape = 0.0
        self.pos_cmd_pub.publish(pos_cmd)

    def load(self):
        rospy.loginfo("In Load stage")
        rospy.sleep(_TSETTLE)  # Does this interfere with hand_state data? Likely not, callbacks are in their own threads
        # ideally something like rospy.spin(_TSETTLE)

        # F_des_pre = self.reader.get_forces_max()  # Method 1
        # self.F_des = np.max(F_des_pre) * np.ones(shape=F_des_pre.shape)  # Method 1
        # self.F_des = self.reader.get_forces_max()  # Method 2
        self.F_des = self.reader.get_forces_max_lofilt() # Method 2.1
        # self.F_des = self.reader.get_forces_max_sum()  # Method 3


        Fmaxes = self.reader.get_forces()  # Method 1 and 2
        # Fmaxes = self.reader.get_forces_sum()  # Method 3, similar to 'AND' on the sensors for contacts

        stable_contact = self.control2(Fmaxes)
        while not stable_contact and not rospy.is_shutdown():
            self.rate.sleep()
            stable_contact = self.control2(Fmaxes)
            # Fmaxes = self.reader.get_forces_max()  # F_des can be determined a priori
            Fmaxes = self.reader.get_forces()

    def lift_hold(self):
        # Need this to happen potentially multiple times while holding and moving
        F1, F2, F3 = self.reader.get_forces()
        F = (F1 + F2 + F3) / 3
        F1_dist, F2_dist, F3_dist = self.reader.get_force_disturbances()
        F_dist = (F1_dist + F2_dist + F3_dist) / 3

        FBP = self.reader.get_forces_bpfilt()
        FBP = np.linalg.norm(FBP)  # Sum or something else here?
        slip = (abs(F_dist) > F * _SLIPTHRESH) and (FBP < _FBPTHRESH)
        if self.replace:
            if slip or (self.reader.get_acc_norm_hifilt() > self.athresh):
                return False
        else:          
            # Still in Lift/Hold phase  
            if slip:
                rospy.loginfo("--Slip detected; adjusting")
                self.F_des *= (
                    _KSLIP  # Current F_des is F_c, carried over from previous step.
                )
        # self.control(self.reader.get_forces_max())
        self.control2(self.reader.get_forces())

        return True

    def lift_hold_loop(self):
        """
        A dummy example of lifting and holding for some amount of time. This will become defunct
        once a real transition method is added.
        """
        rospy.loginfo("In Lift/Hold stage (replace service callable)")
        self.replace = False  # This will get changed when we give the place() command
        # Advertise service that allows us to set self.replace
        self.replace_srv = rospy.Service("/reflex_controller/replace", Trigger, self.set_replace)
        
        while self.lift_hold() and not rospy.is_shutdown():
            self.rate.sleep()
        '''
        # This for loop will be replaced with a while() that waits for a placement command
        s = 1  # minimum/approximate loop duration in seconds
        for i in range(s * _LOOP_HZ):
            if not rospy.is_shutdown():
                holding = self.lift_hold()
                self.rate.sleep()
        '''
        # Once replace is called and controller determines replacement, shut down service
        self.replace_srv.shutdown()

    def lift_hold_dummy(self):
        rospy.loginfo("In Lift/Hold stage (replace service callable)")
        self.replace = False  # This will get changed when we give the place() command
        # Advertise service that allows us to set self.replace
        self.replace_srv = rospy.Service("/reflex_controller/replace", Trigger, self.set_replace)
        while not self.replace and not rospy.is_shutdown():
            self.rate.sleep()
        self.replace_srv.shutdown()

    def set_replace(self, req):
        # Technically once this happens we are in the Replace phase,
        # even though we keep using the lift/hold functions.
        self.replace_calls += 1
        self.replace = True
        self.athresh = _ATHRESH / self.replace_calls
        rospy.loginfo("In Replace stage")  # Acceleration-based by Romano
        return TriggerResponse(success=True)

    def unload(self):
        rospy.loginfo("In Unload stage")
        begin = rospy.get_rostime()
        F_c = np.max(self.reader.get_forces_max())  # F_des can be determined a priori
        while (rospy.get_rostime() - begin).to_sec() < _TUNLOAD and not rospy.is_shutdown():
            self.F_des = F_c * (1 - (rospy.get_rostime() - begin).to_sec() / _TUNLOAD)
            # After computing new desired force, adjust control based on current readings
            self.control2(self.reader.get_forces())
            self.rate.sleep()

    def open_hand(self):
        rospy.loginfo("In Open stage")
        self.v_des = -_VOPEN
        for i in range(10):  # Loop necessary?
            if not rospy.is_shutdown():
                # self.vel_cmd_pub.publish(vel_cmd)
                self.vel_pub()
                # self.control_direct([0, 0, 0]) # Open hand pose
                self.rate.sleep()

    def run(self):
        # TODO do we need explicit threads with the rospy.Timer going on? Will Timer work in Kinetic?
        """
        reader_thread = Thread(target=self.reader.collect)
        reader_thread.start()
        # Cleanup
        rospy.on_shutdown(self.reader.stop)
        rospy.on_shutdown(reader_thread.join)
        """
        self.reader.collect()
        rospy.on_shutdown(self.reader.stop)

        # loop_rate = self.rate  # 1kHz control loop
        functions = [
            self.close_simple,
            # self.load_simple,
            self.load,
            # self.lift_hold_dummy,
            self.lift_hold_loop,
            # self.replace,
            self.unload,
            self.open_hand,
        ]
        # TODO In the future, ROS services and classifiers will determine steps from one function to the next
        # TODO possibly have position/force controllers running in a different thread? Each in their own?
        for i in range(len(functions)):
            if not rospy.is_shutdown():
                functions[i]()
            # loop_rate.sleep()

    def stop(self):
        self.control_direct([0,0,0])
        rospy.loginfo('Shutting down ReFlex controller')


if __name__ == "__main__":
    controller = ReflexController()
    rospy.on_shutdown(controller.stop)
    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass
