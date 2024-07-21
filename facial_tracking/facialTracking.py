import cv2
import time
import serial
import facial_tracking.conf as conf

from facial_tracking.faceMesh import FaceMesh
from facial_tracking.eye import Eye
from facial_tracking.lips import Lips
from facial_tracking.head_pose import Head


class FacialTracker:
    """
    The object of facial tracking, predicting status of eye, iris, and mouth.
    """

    def __init__(self):

        self.fm = FaceMesh()
        self.left_eye = None
        self.right_eye = None
        self.lips = None
        self.head = None
        self.left_eye_closed_frames = 0
        self.right_eye_closed_frames = 0
        self.ser = serial.Serial(
        port='/dev/ttyACM0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
        )
        self.warning_status = 0
        self.warning_status_head = 0
        self.warning_status_yawning = 0

       


    def process_frame(self, frame):
        """Process the frame to analyze facial status."""
        self.detected = False
        self.fm.process_frame(frame)
        self.fm.draw_mesh_lips()

        if self.fm.mesh_result.multi_face_landmarks:
            self.detected = True
            for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                self.left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
                self.right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                self.lips = Lips(frame, face_landmarks, conf.LIPS)
                self.head = Head(frame, face_landmarks, conf.HEAD)
                self._check_eyes_status()
                self._check_yawn_status()
                self._check_head_status()

    def _check_eyes_status(self):
        self.eyes_status = ''

        if self.left_eye.eye_closed():
            self.left_eye_closed_frames += 1
        else:
            self.left_eye_closed_frames = 0
            self.left_eye.iris.draw_iris(True)

        if self.right_eye.eye_closed():
            self.right_eye_closed_frames += 1
        else:
            self.right_eye_closed_frames = 0
            self.right_eye.iris.draw_iris(True)

        if self._left_eye_closed() or self._right_eye_closed():
            self.eyes_status = 'SLEEPING: eye closed'
            self.warning_status = 0
            return

        if not self.left_eye.eye_closed() and not self.right_eye.eye_closed():
            if self.left_eye.gaze_right() and self.right_eye.gaze_right():
                self.eyes_status = 'DISTRACTED: gazing right' 
                self.warning_status = 0
            elif self.left_eye.gaze_left() and self.right_eye.gaze_left():
                self.eyes_status = 'DISTRACTED: gazing left'
                self.warning_status = 0
            elif self.left_eye.gaze_center() and self.right_eye.gaze_center():
                self.eyes_status = 'gazing center'
                self.warning_status = 1

    def _check_yawn_status(self):
        self.yawn_status = ''
        if self.lips.mouth_open():
            self.yawn_status = 'DROWSY: yawning'
            self.warning_status_yawning = 0
        else:
            self.warning_status_yawning = 1

    def _check_head_status(self):
        self.head_status = ''

        if self.head.head_forward():
            self.head_status = 'forward'
            self.warning_status_head = 1
        else:
            if self.head.head_right():
                self.head_status = 'DISTRACTED: looking right'
                self.warning_status_head = 0

                
            elif self.head.head_left():
                self.head_status = 'DISTRACTED: looking left'
                self.warning_status_head = 0            
                
            elif self.head.head_up():
                self.head_status = 'DISTRACTED: looking up'
                self.warning_status_head = 0
                
            elif self.head.head_down():
                self.head_status = 'DISTRACTED: looking down'
                self.warning_status_head = 0
                

    def _left_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.left_eye_closed_frames > threshold

    def _right_eye_closed(self, threshold=conf.FRAME_CLOSED):
        return self.right_eye_closed_frames > threshold


def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    facial_tracker = FacialTracker()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        facial_tracker.process_frame(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.6,
                    conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)

        if facial_tracker.detected:
            cv2.putText(frame, f'{facial_tracker.eyes_status}', (30, 70), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{facial_tracker.yawn_status}', (30, 150), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, f'{facial_tracker.head_status}', (30, 110), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Facial tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()