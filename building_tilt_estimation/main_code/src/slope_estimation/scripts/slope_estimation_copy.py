#!/usr/bin/env python

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
import message_filters
from message_filters import TimeSynchronizer, Subscriber
import math


class DataProcessor:
    def __init__(self):
        self.latest_filtered_scan_data = None
        self.latest_odom_data = None
        self. count = 0
        self.stored_scans = []
        self.stored_positions = []
        rospy.set_param('get_slope', False)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.filtered_scan_sub = rospy.Subscriber('/filtered_scan', LaserScan, self.filtered_scan_callback)
        self.odom_sub = rospy.Subscriber('/mavros/global_position/local', Odometry, self.odom_callback)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
    
        self.filtered_scan_pub = rospy.Publisher('/filtered_scan', LaserScan, queue_size=10)

        # Create subscribers for laser scan and odometry topics
        laser_sub = Subscriber('/filtered_scan', LaserScan)
        odom_sub = Subscriber('/mavros/global_position/local', Odometry)
    
        # Create a time synchronizer for laser scan and odometry data
        # ts = TimeSynchronizer([laser_sub, odom_sub], queue_size=10)
        ts = message_filters.ApproximateTimeSynchronizer([laser_sub, odom_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.synchronized_callback)


    def synchronized_callback(self, scan_data, odom_data):
        # This function will be called when synchronized data is available
        # Process synchronized scan and odom data here
        # print("synchronized data")
        # print("Synchronized Scan ranges length:", len(scan_data.ranges))
        # print("Synchronized Odometry pose:", odom_data.pose.pose)
        if rospy.get_param('get_slope'):
            if (self.count == 0 or self.count == 100 ):
                self.stored_scans.append(scan_data.ranges)
                self.stored_positions.append(odom_data.pose.pose.position)
    
                if self.count == 100 : 
                    range_difference = self.stored_scans[-1][0] - self.stored_scans[-2][0]
                    distance_difference = self.stored_positions[-1].x - self.stored_positions[-2].x
                    slope = range_difference / distance_difference
                    print ("slope of the given data is : ", slope)
                    print("the two ranges are : ", self.stored_scans[-1][0], " " , self.stored_scans[-2][0], " ", range_difference)
                    print("the two positoins values are : ", self.stored_positions[-1].x , "  ", self.stored_positions[-2].x)
                    self.count = 0
                    rospy.set_param('get_slope', False)
                self.count += 1
            else:
                self.count += 1 
        print(self.count)

    def odom_callback(self, odom_msg):
        # This function will be called whenever new odometry data is received
        
        # Create a TransformStamped message
        transform_stamped = TransformStamped()
        transform_stamped.header = odom_msg.header
        transform_stamped.child_frame_id = "base_link"  # Change this to match your setup
        transform_stamped.transform.translation = odom_msg.pose.pose.position
        transform_stamped.transform.rotation = odom_msg.pose.pose.orientation
        
        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_stamped)
    
    def filtered_scan_callback(self, filtered_scan_data):
        self.latest_filtered_scan_data = filtered_scan_data
        

    def scan_callback(self, scan_data):

        # Define the range of angle to get filtered data
        desired_angle_min_deg = -97
        desired_angle_max_deg =  -83
        desired_angle_min_rad = math.radians(desired_angle_min_deg)
        desired_angle_max_rad = math.radians(desired_angle_max_deg)

        start_idx = int((desired_angle_min_rad - scan_data.angle_min) / scan_data.angle_increment)
        end_idx = int((desired_angle_max_rad - scan_data.angle_min) / scan_data.angle_increment)

        # This function will be called whenever new laser scan data is received
        # Do something with the scan data here
        # For example, print the ranges of the laser scan
        # print("Laser Scan Ranges:", len(scan_data.ranges))
        filtered_scan = LaserScan()
        filtered_scan.header = scan_data.header
        filtered_scan.angle_min = desired_angle_min_rad
        filtered_scan.angle_max = desired_angle_max_rad
        filtered_scan.angle_increment = scan_data.angle_increment
        filtered_scan.time_increment = scan_data.time_increment
        filtered_scan.scan_time = scan_data.scan_time
        filtered_scan.range_min = scan_data.range_min
        filtered_scan.range_max = scan_data.range_max
        
        # Modify the scan data, changing 'inf' values to 2
        filtered_scan.ranges = [float('inf') if value == float('inf') else value for value in scan_data.ranges[start_idx:end_idx + 1]]
        # filtered_scan.ranges = scan_data.ranges[start_idx:end_idx + 1]
        
        # Publish the filtered scan data on a new topic
        self.filtered_scan_pub.publish(filtered_scan)


if __name__ == '__main__':
    rospy.init_node('laser_subscriber', anonymous=True)
    process_data = DataProcessor()
    rospy.spin()