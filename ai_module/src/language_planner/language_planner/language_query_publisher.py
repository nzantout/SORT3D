import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LanguagePublisher(Node):
    def __init__(self):
        super().__init__('language_publisher')
        self.publisher_ = self.create_publisher(String, '/language_planner_query', 10)
        self.timer_ = self.create_timer(0.1, self.publish_input)
        self.get_logger().info("LanguagePublisher node has been started. Type your query below.")

    def publish_input(self):
        user_input = input("Enter a query to publish: ")
        if user_input:
            msg = String()
            msg.data = user_input
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published: '{user_input}'")

def main():
    rclpy.init()

    node = LanguagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
