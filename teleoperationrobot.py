import time
import threading
import cv2
import mediapipe as mp
import math
import numpy as np
import argparse
import utilities

from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2


class RobotActions:
    def __init__(self, router, proportional_gain=2.0):
        self.proportional_gain = proportional_gain
        self.router = router
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(router)

    @staticmethod
    def check_for_end_or_abort(e):
        def check(notification, error=e):
            if notification.action_event == Base_pb2.ACTION_END or notification.action_event == Base_pb2.ACTION_ABORT:
                error.set()

        return check

     # Comandos do gripper
    def gripper_commands(self, percent):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        position = 1 - percent

        finger.finger_identifier = 1
        finger.value = position
        self.base.SendGripperCommand(gripper_command)
        time.sleep(0.034)

    def move_to_workspace(self):
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        # Place arm straight up
        i = 0
        list_joints = [81.85, 326.15, 125.05, 85.26, 291.24, 271.04]
        for joint_pose in list_joints:
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = i
            joint_angle.value = joint_pose  # 125
            i += 1

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)

        finished = e.wait(20)
        self.base.Unsubscribe(notification_handle)

        return finished

    def move_cartesian(self, X, Y):
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()
        cartesian_pose = action.reach_pose.target_pose

        cartesian_pose.x = X
        cartesian_pose.y = feedback.base.tool_pose_y
        cartesian_pose.z = Y
        cartesian_pose.theta_x = 90.6  # 89.3
        cartesian_pose.theta_y = -1  # 0
        cartesian_pose.theta_z = 150  # 177.2

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)

        finished = e.wait(20)
        self.base.Unsubscribe(notification_handle)

        return finished


def interpolate_coordinates(input_x, input_y):
    x = int(np.interp(int(input_x), [0, 640], [-420, 420 - 1]))
    z = int(np.interp(int(input_y), [0, 480], [420 - 1, -8]))

    # Transform to centimeters
    x = x / 1000
    z = z / 1000

    return x, z


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = utilities.parseConnectionArguments(parser)

    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        robot_instance = RobotActions(router)
        robot_instance.move_to_workspace()

        # solution APIs inicio do codigo medindo distancia dos pontos
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        percent_bar, gripper_percent = 400, 0

        # Webcam Setup
        cam = cv2.VideoCapture(0)

        # Mediapipe Hand Landmark Model
        with mp_hands.Hands(
                model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cam.isOpened():
                success, image = cam.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                landmarks_list = []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                if results.multi_hand_landmarks:
                    my_hand = results.multi_hand_landmarks[0]
                    for ids, lm in enumerate(my_hand.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks_list.append([ids, cx, cy])

                # Assigning variables for Thumb and Index finger position
                if len(landmarks_list) != 0:
                    x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
                    x2, y2 = landmarks_list[8][1], landmarks_list[8][2]

                    # Marking Thumb and Index finger
                    cv2.circle(image, (x1, y1), 15, (255, 255, 255))
                    cv2.circle(image, (x2, y2), 15, (255, 255, 255))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    length = math.hypot(x2 - x1, y2 - y1)
                    if length < 50:
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    percent_bar = np.interp(length, [50, 220], [400, 150])
                    gripper_percent = np.interp(length, [50, 220], [0, 100])

                    # Gripper Bar
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(image, (50, int(percent_bar)), (85, 400), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'{int(gripper_percent)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0),
                                3)

                    # Chama a função de movimentação a partir desse ponto
                    workspace_x, workspace_z = interpolate_coordinates(landmarks_list[9][1], landmarks_list[9][2])
                    robot_instance.move_cartesian(workspace_x, workspace_z)

                if int(gripper_percent) == 0:
                    robot_instance.gripper_commands(gripper_percent / 100)
                else:
                    robot_instance.gripper_commands(gripper_percent / 100)

                image = cv2.flip(image, 1)
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cam.release()


if __name__ == "__main__":
    main()
