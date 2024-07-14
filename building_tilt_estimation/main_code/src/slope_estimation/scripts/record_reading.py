#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Int16

def trigger():
    pub = rospy.Publisher('record_trigger', Int16, queue_size=10)
    rospy.init_node('record_trigger_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    trigger_value = 0
    count = 0
    rospy.set_param('reading_trigger', False)

    while not rospy.is_shutdown():
        get_trigger = rospy.get_param("reading_trigger")
        if(count <= 10 and get_trigger == True):
            if(count == 10):
                count = 0
                rospy.set_param('reading_trigger', False)
                trigger_value = 0
                continue
            count += 1
            trigger_value = 100
        rospy.loginfo(trigger_value)
        pub.publish(trigger_value)
        rate.sleep()

if __name__ == '__main__':
    try:
        trigger()
    except rospy.ROSInterruptException:
        pass
