#!/usr/bin/env python
import rospy
from std_srvs.srv import Empty, Trigger, TriggerResponse
from std_msgs.msg import Bool
from reflex_msgs2.msg import PoseCommand, VelocityCommand, Hand


class ReflexInterface2:

    def __init__(self):
        # Topics
        self.hand_state_topic   = "/reflex_takktile2/hand_state"
        self.pos_cmd_topic      = "/reflex_takktile2/command_position"
        self.vel_cmd_topic      = "/reflex_takktile2/command_velocity"
        self.grasp_status_topic = "/reflex_interface2/grasp_status"
        # External services this class calls
        self.zero_tactile_srv          = "/reflex_takktile2/calibrate_tactile"
        self.calibrate_srv             = "/reflex_takktile2/calibrate_fingers"
        self.enable_tactile_stops_srv  = "/reflex_takktile2/enable_tactile_stops"
        self.disable_tactile_stops_srv = "/reflex_takktile2/disable_tactile_stops"
        # Internal services this class offers
        open_hand_srv = "/reflex_interface2/open_hand"
        grasp_srv     = "/reflex_interface2/grasp"

        self.pos_cmd_pub = rospy.Publisher(self.pos_cmd_topic, PoseCommand, queue_size=1)
        self.vel_cmd_pub = rospy.Publisher(self.vel_cmd_topic, VelocityCommand, queue_size=1)
        self.status_pub  = rospy.Publisher(self.grasp_status_topic, Bool, queue_size=1)
        rospy.Subscriber(self.hand_state_topic, Hand, self._reflex_state_cb)
        self.open_hand_srv = rospy.Service(open_hand_srv, Trigger, self.open_hand)
        self.grasp_srv     = rospy.Service(grasp_srv, Trigger, self.grasp)

        self.rate = rospy.Rate(100)
        self.hand_state = None

        # TODO this is a huge hack just to monitor when Finger 2 goes bad. There is a firmware
        # issue such that at some point the tactile sensors just stop reading and the state
        # becomes fixed on some bogus values. Thus we keep a rolling queue of the sum of the
        # pressure values of that finger. There is always sensor noise, so if the finger is
        # functioning you'll get small fluctuations in the pressures and the sum of the
        # pressures therefore fluctuates. But, if it blows out, the values are fixed over the
        # full queue window and we know it needs to be powered off and on again until it's
        # resolved. Ideally the firmware issue can be fixed somehow.
        self.finger_2_is_blown = False
        self.finger_2_state_queue = range(100)
        

    def run(self):
        rospy.loginfo("Waiting for external services...")
        rospy.wait_for_service(self.zero_tactile_srv)
        rospy.wait_for_service(self.calibrate_srv)
        rospy.wait_for_service(self.enable_tactile_stops_srv)
        rospy.loginfo("Services are up!")

        # self._zero_tactile()
        # self._calibrate_fingers()
        self._enable_tactile_stops()
        
        rospy.loginfo("Waiting for hand state...")
        while not rospy.is_shutdown() and not self.hand_state:
            self.rate.sleep()
        rospy.loginfo("Hand state received!")

        rospy.loginfo("ReFlex hand interface services are available.")
        msg_shown = False
        while not rospy.is_shutdown():
            # Tell user when finger is messed up and needs to re-power
            if self.finger_2_is_blown and not msg_shown:
                rospy.logerr("ReFlex Finger is done. Need to power down and restart.")
                msg_shown = True
            self.rate.sleep()

    def stop(self):
        rospy.loginfo("Exiting.")

        
    # === BEGIN Service functions this class offers ===============================================
        
    def open_hand(self, req):
        pos_cmd = PoseCommand()
        pos_cmd.f1 = 0
        pos_cmd.f2 = 0
        pos_cmd.f3 = 0
        pos_cmd.preshape = 0.0
        for i in range(10):
            self.pos_cmd_pub.publish(pos_cmd)
        self.grasp_status = False
        return TriggerResponse(success=True)

    def grasp(self, req):
        if self.finger_2_is_blown:
            rospy.logwarn("Cannot execute grasp: Finger2 is not working.")
            return TriggerResponse(success=False)
        
        self._close_hand()
        # Wait until all fingers are moving
        while self._all_fingers_stopped():
            self.rate.sleep()
        # Wait until they are all stopped
        while not self._all_fingers_stopped():
            self.rate.sleep()
        self._tighten_hand()

        return TriggerResponse(success=True)
        
    # === END Service functions this class offers =================================================


    def _reflex_state_cb(self, hand_state):
        self.hand_state = hand_state
        self.finger_2_state_queue.append(sum(self.hand_state.finger[1].pressure))
        self.finger_2_state_queue.pop(0)
        if all(x == self.finger_2_state_queue[0] for x in self.finger_2_state_queue):
            self.finger_2_is_blown = True
    
    def _zero_tactile(self):
        try:
            zero_tactile = rospy.ServiceProxy(self.zero_tactile_srv, Empty)
            zero_tactile()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to zero ReFlex tactile failed: %s" % e)
            return False       
        return True

    def _calibrate_fingers(self):
        try:
            calibrate = rospy.ServiceProxy(self.calibrate_srv, Empty)
            calibrate()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to calibrate fingers failed: %s" % e)
            return False
        return True

    def _enable_tactile_stops(self):
        try:
            enable_stops = rospy.ServiceProxy(self.enable_tactile_stops_srv, Empty)
            enable_stops()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to enable tactile stops failed: %s" % e)
            return False
        return True

    def _disable_tactile_stops(self):
        try:
            disable_stops = rospy.ServiceProxy(self.disable_tactile_stops_srv, Empty)
            disable_stops()
        except rospy.ServiceException as e:
            rospy.logwarn("Service request to disable tactile stops failed: %s" % e)
            return False
        return True

    def _all_fingers_in_contact(self):
        f1_in_contact = sum(self.hand_state.finger[0].contact) > 0
        f2_in_contact = sum(self.hand_state.finger[1].contact) > 0
        f3_in_contact = sum(self.hand_state.finger[2].contact) > 0
        return f1_in_contact and f2_in_contact and f3_in_contact

    def _all_fingers_stopped(self, stop_threshold=10.0):
        f1_stopped = self.hand_state.motor[0].velocity < stop_threshold
        f2_stopped = self.hand_state.motor[1].velocity < stop_threshold
        f3_stopped = self.hand_state.motor[2].velocity < stop_threshold
        return f1_stopped and f2_stopped and f3_stopped

    def _close_hand(self):
        self._zero_tactile()
        self._enable_tactile_stops()
        # Close the hand at constant velocity until it contacts object
        vel_cmd = VelocityCommand()
        vel_cmd.f1 = 1.0
        vel_cmd.f2 = 1.0  # bad finger
        vel_cmd.f3 = 1.0
        vel_cmd.preshape = 0.0
        for i in range(10):
            self.vel_cmd_pub.publish(vel_cmd)

    def _tighten_hand(self, pos_increment=0.8):
        self._disable_tactile_stops()
        pos_cmd = PoseCommand()
        pos_cmd.f1 = self.hand_state.finger[0].proximal + pos_increment
        pos_cmd.f2 = self.hand_state.finger[1].proximal + pos_increment
        pos_cmd.f3 = self.hand_state.finger[2].proximal + pos_increment
        pos_cmd.preshape = 0.0
        for i in range(10):
            self.pos_cmd_pub.publish(pos_cmd)
    
    
if __name__ == '__main__':
    rospy.init_node("reflex_interface2")
    try:
        interface = ReflexInterface2()
        rospy.on_shutdown(interface.stop)
        interface.run()
    except rospy.ROSInterruptException:
        pass
        
       
