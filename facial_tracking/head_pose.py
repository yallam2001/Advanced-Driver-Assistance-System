import cv2
import time
import facial_tracking.conf as conf
import numpy as np

from facial_tracking.faceMesh import FaceMesh


class Head:
    """
        The object of head, computing its features from face landmarks.

        Args:
            frame (numpy,ndarray): the input frame
            face_landmarks (mediapipe face landmarks object): contains the face landmarks coordinates
            id (list of int): the indices of eye in the landmarks
    """

    def __init__(self, frame, face_landmarks, id):
        self.frame = frame
        self.face_landmarks = face_landmarks
        self.id = id

        self.pos_2d, self.pos_3d = self._get_head_pos()
        self.pos_relative = self._get_head_angles()

    def _get_head_pos(self):
        h, w = self.frame.shape[:2]
        head_pos_2d = []
        head_pos_3d = []
        for id in self.id:
            pos = self.face_landmarks.landmark[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            cz = pos.z
            head_pos_2d.append([cx, cy])
            head_pos_3d.append([cx, cy, cz])

        head_pos_2d = np.array(head_pos_2d, dtype=np.float64)
        head_pos_3d = np.array(head_pos_3d, dtype=np.float64)

        return head_pos_2d, head_pos_3d

    def _get_head_angles(self):
        h, w = self.frame.shape[:2]
        focal_length = 1 * w

        cam_matrix = np.array([[focal_length, 0, h / 2],
                               [0, focal_length, w / 2],
                               [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(self.pos_3d, self.pos_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        return [x, y, z]

    def head_left(self, threshold=conf.HEAD_LEFT):
        """Check whether head left."""
        return self.pos_relative[1] > threshold

    def head_right(self, threshold=conf.HEAD_RIGHT):
        """Check whether head right."""
        return self.pos_relative[1] < threshold

    def head_up(self, threshold=conf.HEAD_UP):
        """Check whether head left."""
        return self.pos_relative[0] > threshold

    def head_down(self, threshold=conf.HEAD_DOWN):
        """Check whether head right."""
        return self.pos_relative[0] < threshold

    def head_forward(self):
        """Check whether head forward."""
        return not self.head_left() and not self.head_right() and not self.head_up() and not self.head_down()


def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    fm = FaceMesh()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        text = 'unkown'

        fm.process_frame(frame)
        #fm.draw_mesh()
        if fm.mesh_result.multi_face_landmarks:
            for face_landmarks in fm.mesh_result.multi_face_landmarks:
                head = Head(frame, face_landmarks, conf.HEAD)

                if head.head_forward():
                    text = 'forward'
                else:
                    if head.head_right():
                        text = 'looking right'
                    elif head.head_left():
                        text = 'looking left'
                    elif head.head_up():
                        text = 'looking up'
                    elif head.head_down():
                        text = 'looking down'

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30, 30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'{text}', (30, 70), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Head Pose Estimation', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
