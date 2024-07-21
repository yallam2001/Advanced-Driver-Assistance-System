import numpy as np

import cv2

from tflite_runtime.interpreter import Interpreter

import time

import serial

ser = serial.Serial(

        port='/dev/ttyACM0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0

        baudrate = 9600,

        parity=serial.PARITY_NONE,

        stopbits=serial.STOPBITS_ONE,

        bytesize=serial.EIGHTBITS

        )

#############################################

threshold = 0.95  # PROBABILITY THRESHOLD

font = cv2.FONT_HERSHEY_SIMPLEX 

##############################################

# SET UP THE VIDEO CAMERA

cap = cv2.VideoCapture(0)

# Load the TFLite model and allocate tensors.

interpreter = Interpreter(model_path="/home/pi/Newfolder/my_neural_network_model.tflite")

interpreter.allocate_tensors()

# Get input and output tensors.

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()

prev_time = time.time()


def detect_lane(frame):
    # Define region of interest (ROI)



    height, width = frame.shape[:2]



    roi = frame[int(height / 2):height, :]



    # Convert ROI to grayscale



    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



    # Perform Gaussian blur to reduce noise



    blurred = cv2.GaussianBlur(gray, (5, 5), 0)



    # Apply Canny edge detection



    edges = cv2.Canny(blurred, 50, 150)



    # Perform probabilistic Hough transform



    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=50)



    # Filter out lane lines



    left_lane_lines = []




    right_lane_lines = []



    if lines is not None:



        for line in lines:



            x1, y1, x2, y2 = line[0]



            slope = (y2 - y1) / (x2 - x1)



            if slope < -0.5 and slope > -5:  # Left lane line



                left_lane_lines.append(line)



            elif slope > 0.5 and slope < 5:  # Right lane line



                right_lane_lines.append(line)



    # Draw left lane lines



    left_detected = False



    turn_right = 0



    if left_lane_lines:



        left_detected = True



        left_lane_avg = np.mean(left_lane_lines, axis=0, dtype=np.int32)



        x1, y1, x2, y2 = left_lane_avg[0]

        # print ('x2 =', x2)

        if x2 > 700:  # turn right



            turn_right = 1



        cv2.line(frame, (x1, y1 + int(height / 2)), (x2, y2 + int(height / 2)), (0, 255, 0), 2)



    # Draw right lane lines



    right_detected = False



    turn_left = 0



    if right_lane_lines:



        right_detected = True



        right_lane_avg = np.mean(right_lane_lines, axis=0, dtype=np.int32)



        x1, y1, x2, y2 = right_lane_avg[0]

        # print ('x1 =', x1)

        if x1 < 700:  # turn left



            turn_left = 1



        cv2.line(frame, (x1, y1 + int(height / 2)), (x2, y2 + int(height / 2)), (0, 255, 0), 2)



    # Draw the area between the lanes as a green polygon



    if left_detected and right_detected:



        # Extract x coordinates of the left and right lane lines



        left_x = left_lane_avg[0][0]



        right_x = right_lane_avg[0][2]




    if left_detected and turn_right:



        cv2.putText(frame, 'Turn Right', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        char_to_send = 'R'  # Replace with the character you want to send



        ser.write(char_to_send.encode('utf-8'))



    if right_detected and turn_left:



        cv2.putText(frame, 'Turn Left', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



        char_to_send = 'L'  # Replace with the character you want to send



        ser.write(char_to_send.encode('utf-8'))





    # Display detection status



    if left_detected and right_detected:



        if not turn_right and not turn_left:



            cv2.putText(frame, 'Keep Straight', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



            char_to_send = 'A'  # Replace with the character you want to send



            ser.write(char_to_send.encode('utf-8'))



        cv2.putText(frame, 'Lanes Detected', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    else:



        cv2.putText(frame, 'Lanes Not Detected', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



    return frame


def preprocess_image(img):

    # Convert to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return thresh


def grayscale(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 


def equalize(img):

    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):

    img = grayscale(img)

    img = equalize(img)

    img = cv2.resize(img, (32, 32))  # Resize to match model input size

    img = img / 255

    return img


def getClassName(classNo):

    if classNo == 0:

        return 'Speed Limit 20 km/h'

    elif classNo == 1:

        return 'Speed Limit 30 km/h'

    elif classNo == 2:

        return 'Speed Limit 50 km/h'

    elif classNo == 3:

        return 'Speed Limit 60 km/h'

    elif classNo == 4:

        return 'Speed Limit 70 km/h'

    elif classNo == 5:

        return 'Speed Limit 80 km/h'

    elif classNo == 6:

        return 'End of Speed Limit 80 km/h'

    elif classNo == 7:

        return 'Speed Limit 100 km/h'

    elif classNo == 8:

        return 'Speed Limit 120 km/h'

    elif classNo == 9:

        return 'No passing'

    elif classNo == 10:

        return 'No passing for vehicles over 3.5 metric tons'

    elif classNo == 11:

        return 'Right-of-way at the next intersection'

    elif classNo == 12:

        return 'Priority road'

    elif classNo == 13:

        return 'Yield'

    elif classNo == 14:

        return 'Stop'

    elif classNo == 15:

        return 'No vehicles'

    elif classNo == 16:

        return 'Vehicles over 3.5 metric tons prohibited'

    elif classNo == 17:

        return 'No entry'

    elif classNo == 18:

        return 'General caution'

    elif classNo == 19:

        return 'Dangerous curve to the left'

    elif classNo == 20:

        return 'Dangerous curve to the right'

    elif classNo == 21:

        return 'Double curve'

    elif classNo == 22:

        return 'Bumpy road'

    elif classNo == 23:

        return 'Slippery road'

    elif classNo == 24:

        return 'Road narrows on the right'

    elif classNo == 25:

        return 'Road work'

    elif classNo == 26:

        return 'Traffic signals'

    elif classNo == 27:

        return 'Pedestrians'

    elif classNo == 28:

        return 'Children crossing'

    elif classNo == 29:

        return 'Bicycles crossing'

    elif classNo == 30:

        return 'Beware of ice/snow'

    elif classNo == 31:

        return 'Wild animals crossing'

    elif classNo == 32:

        return 'End of all speed and passing limits'

    elif classNo == 33:

        return 'Turn right ahead'

    elif classNo == 34:

        return 'Turn left ahead'

    elif classNo == 35:

        return 'Ahead only'

    elif classNo == 36:

        return 'Go straight or right'

    elif classNo == 37:

        return 'Go straight or left'

    elif classNo == 38:

        return 'Keep right'

    elif classNo == 39:

        return 'Keep left'

    elif classNo == 40:

        return 'Roundabout mandatory'

    elif classNo == 41:

        return 'End of no passing'

    elif classNo == 42:

        return 'End of no passing by vehicles over 3.5 metric tons'


while True:

    # READ IMAGE    
    success, imgOriginal = cap.read()

    height, width = imgOriginal.shape[:2]

    if not success:

        break

    imgOriginal = detect_lane(imgOriginal)

    gray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect circles using Hough Circle Transform

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=50, param2=30, minRadius=55, maxRadius=75)

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        # Take only the first detected circle

        x, y, r = circles[0]

        # Define the ROI around the detected circle

        roi_left = max(0, x - r-5)

        roi_right = min(width, x + r+5)

        roi_top = max(0, y - r-5)

        roi_bottom = min(height, y + r+5)

        roi = imgOriginal[roi_top:roi_bottom, roi_left:roi_right]

        # PROCESS IMAGE

        img = np.asarray(roi)

        img = preprocessing(img)

        cv2.imshow('processed image', img)

        img = img.astype(np.float32)  # Convert to FLOAT32

        img = img.reshape(1, 32, 32, 1)

        # PREDICT IMAGE

        # Set the tensor to point to the input data

        interpreter.set_tensor(input_details[0]['index'], img)

        # Run the model

        interpreter.invoke()

        # Get the results

        predictions = interpreter.get_tensor(output_details[0]['index'])

        probabilityValue = np.amax(predictions)

        classIndex = np.argmax(predictions)

        if probabilityValue > threshold:

            class_name = getClassName(classIndex)

            cv2.putText(imgOriginal, class_name, (roi_left + 20, roi_top + 35), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (roi_left + 20, roi_top + 75), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            if classIndex == 1:
                print('g')
                char_to_send = 'F'  # Replace with the character you want to send

                ser.write(char_to_send.encode('utf-8'))

            if classIndex == 0:
                print('y')
                char_to_send = 'S'  # Replace with the character you want to send

                ser.write(char_to_send.encode('utf-8'))


            if classIndex == 14:
                print('s')
                char_to_send = 'P'  # Replace with the character you want to send

                ser.write(char_to_send.encode('utf-8'))

    # Calculate FPS

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

     # Display FPS on the frame

    cv2.putText(imgOriginal, f"FPS: {fps:.2f}", (1100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Resize the frame to fit on screen

    imgOriginal = cv2.resize(imgOriginal, (width // 3, height // 3))     

    cv2.imshow('Result', imgOriginal)

    if cv2.waitKey(1) and 0xFF == ord('q'):

        break

# Release the camera and close all OpenCV windows

cap.release()

cv2.destroyAllWindows()