import rospy
from std_msgs.msg import String


class LanguagePublisher:
    def __init__(self):
        rospy.init_node('language_publisher')
        self.publisher_ = rospy.Publisher('/language_planner_query', String, queue_size=10)
        self.timer_ = rospy.Timer(rospy.Duration(0.1), self.publish_input)
        rospy.loginfo("LanguagePublisher node has been started. Type your query below.")

    def publish_input(self, event):
        user_input = input("Enter a query to publish: ")
        if user_input:
            msg = String()
            msg.data = user_input
            self.publisher_.publish(msg)
            rospy.loginfo(f"Published: '{user_input}'")

def main():

    node = LanguagePublisher()

    rospy.spin()


if __name__ == '__main__':
    main()
