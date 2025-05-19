import numpy as np
import cv2
import os

def red_pass(image):
    """
    Apply a red pass filter to an image.
    """
    # Convert to HSV color space
    #normalize the image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)



    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define red color range
    lower_red1 = np.array([0, 120, 70])   # Lower bound of the first red range
    upper_red1 = np.array([10, 255, 255]) # Upper bound of the first red range
    lower_red2 = np.array([170, 120, 70]) # Lower bound of the second red range
    upper_red2 = np.array([180, 255, 255]) # Upper bound of the second red range

    # Create masks for red
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original image
    red_filtered = cv2.bitwise_and(image, image, mask=red_mask)
    #inc
    # Optionally save the result
    #increase the brightness of the red color
    red_filtered = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2HSV)
    red_filtered[:,:,2] += 128
    #back to bgr
    
    red_filtered = cv2.cvtColor(red_filtered, cv2.COLOR_HSV2BGR)

    #cv2.imwrite('red_filtered.jpg', red_filtered)
    return red_filtered
def simple_sobel(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #Apply Gaussian Blur
    gray = cv2.GaussianBlur(gray, (5, 5),  5)

    # Compute the gradients in the x and y directions
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    # Compute the magnitude of the gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    # Normalize the magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    #threshold the magnitude
    threshold = 40
    magnitude[magnitude > threshold] = 255
    magnitude[magnitude <= threshold] = 0
    cv2.imwrite('magnitude.jpg', magnitude)

    return magnitude, direction


def readGroundtruth(image_name, frame, filename='groundtruth.txt'):
    groundtruth = []
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            #find corresponding image 
            if img_name == image_name[image_name.rfind('NoEntry') : image_name.rfind('.')]:
                x = float(content_list[1])
                y = float(content_list[2])
                width = float(content_list[3])
                height = float(content_list[4])
                #draw bounding box for ground truth
                start_point = (int(x), int(y))
                end_point = (int(x + width), int(y + height))
                colour = (0, 0, 255)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
                groundtruth.append([x, y, x + width, y + height])
    return groundtruth
def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xH = max(boxA[0], boxB[0])
    yH = max(boxA[1], boxB[1])
    xL = min(boxA[2], boxB[2])
    yL = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xL - xH) * max(0, yL - yH)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou 

#function to calculate the F1 score
def F1_score (detected, groundtruth):
    TP = 0
    FP = 0
    FN = 0
    done = []
    for j in range(0, len(groundtruth)):
        for i in range(0, len(detected)):
            if IOU(detected[i], groundtruth[j]) >= 0.5 and detected[i] not in done:
                TP += 1
                done.append(detected[i])
                break
                
    for i in range(0, len(detected)):
        if detected[i] not in done:
            FP += 1 
    FN = len(groundtruth) - TP
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    F1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) != 0 else 0
    return TPR, F1 

def hough_space(accumulator):
    #visualises a 2d hough space
    hough_space1 = accumulator
    hough_space1 = np.sum(hough_space1, axis=2)
    cv2.imwrite('hough_space.jpg', hough_space1)
    
    
    return hough_space

#function to check if the circle can fit in the bounding box
def square_circle(circle, box):
    #bounding boxes are squares so we can use the height or width to check if the circle can fit in the bounding box
    s_width = box[2] - box[0]
    s_height = box[3] - box[1]
    radius = circle[2]
    #check if the circle can fit in the bounding box with a 20% margin of error and make sure the circle is not too small
    if radius*2 <= s_height*1.2 and radius*2>=s_height*0.8:
        return True
    return False

#the function that combines the viola jones and hough transform
def hough_vj(hough_detected, vj_detected, frame):
    true_sign = []
    for box in vj_detected:
        #its a square so you can use either width or height interchangeably
        width = box[2] - box[0]
        height = box[3] - box[1]
        for (x, y, r) in hough_detected:
            #check if the circle centre is in the bounding box's circle region
            if x > box[0]+width//4 and y > box[1]+width//4 and x < box[2]-width//4 and y < box[3] - height//4:
                #checks if the circle can fit the bounding box
                if square_circle((x, y, r), box): 
                    start_point = (box[0], box[1])
                    end_point = (box[2], box[3])
                    colour = (0, 255, 0)
                    thickness = 2
                    frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
                    true_sign.append(box)
                    #cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    break


    return true_sign

def detect_noentry(frame, minNeighbors=1, minSize=(30, 30)):
    #train the viola jones cascade classifier
    detected = []
    noentry_cascade = cv2.CascadeClassifier('NoEntrycascade/cascade.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    noentry = noentry_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=minNeighbors, flags=0, minSize=(30, 30), maxSize=(500,500))

    for i in range(len(noentry)):
        start_point = (noentry[i][0], noentry[i][1])
        end_point = (noentry[i][0] + noentry[i][2], noentry[i][1] + noentry[i][3])
        detected.append([start_point[0], start_point[1], end_point[0], end_point[1]])
    return detected


def hough_circles(frame, image, direction, radius_range, threshold=7, deg_freedom=1):
    #custom hough transform
    height, width = image.shape
    accumulator = np.zeros((height, width, len(radius_range)), dtype=np.int32)
    edge_points = np.argwhere(image > 0)
    for radius_idx, radius in enumerate(radius_range):
        for y, x in edge_points:
            #angle of edge pixel from the direction image
            angle = direction[y][x]
            #if needed you could give the function a degree of freedom 
            # after emperical testing it increased computational time so kept it at 1
            for delta in range(0, deg_freedom, 1):  # 1-degree steps
                angle_rad = angle+ np.deg2rad(delta)
                a = int(x + radius * np.cos(angle_rad))
                b = int(y + radius * np.sin(angle_rad))

                if 0 <= a < width and 0 <= b < height:
                    accumulator[b, a, radius_idx] += 1

    circles = []

    for radius_idx, radius in enumerate(radius_range):
        for y, x in zip(*np.where(accumulator[:, :, radius_idx] >= threshold)):
            circles.append((x, y, radius))
            
    
    return circles, accumulator


def subtask3(filename):
    frame = cv2.imread(filename, 1)
    red = red_pass(frame)
    vj_detected = detect_noentry(red, minNeighbors = 1)
    signs = vj_detected
    groundtruth = readGroundtruth(filename, frame)
    TPR, F1 = F1_score(signs, groundtruth)
    print(f'TPR: {TPR}, F1: {F1}')

    for sign in signs:
        start_point = (sign[0], sign[1])
        end_point = (sign[2], sign[3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    cv2.imwrite('detected.jpg', frame)

    return signs

def subtask2(filename):
    frame = cv2.imread(filename, 1)
    vj_detected = detect_noentry(frame, minNeighbors = 1)
    mag, direction = simple_sobel(frame)
    #the radius range for the hough transform
    rd = range(20, 100, 1)

    hough_detected, accumulator = hough_circles(frame, mag, direction, rd, 7, deg_freedom=1)
    hough_space(accumulator)
    signs = hough_vj(hough_detected, vj_detected, frame)
    groundtruth = readGroundtruth(f'{filename}', frame)
    TPR, F1 = F1_score(signs, groundtruth)
    print(f'TPR: {TPR}, F1: {F1}')
    for sign in signs:
        start_point = (sign[0], sign[1])
        end_point = (sign[2], sign[3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    cv2.imwrite('detected.jpg', frame)
    
def subtask1(filename):
    frame = cv2.imread(filename, 1)
    signs = detect_noentry(frame, minNeighbors=3)
    groundtruth = readGroundtruth(f'{filename}', frame)
    TPR, F1 = F1_score(signs, groundtruth)
    print(f'TPR: {TPR}, F1: {F1}')
    for sign in signs:
        start_point = (sign[0], sign[1])
        end_point = (sign[2], sign[3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
    cv2.imwrite('detected.jpg', frame)
import sys

# Check if filename was provided
if len(sys.argv) < 2:
    print("Usage: python program.py <filename>, please enter an image path")
    sys.exit(1)  # Exit the program with an error code

# Get filename from the command-line arguments
file_name = sys.argv[1]
print(f"You provided the filename: {file_name}")
if __name__ == '__main__':
    #uncomment for only the viola jones detector
    #subtask1(file_name)
    #uncomment for subtask 2 where I only use the hough and viola jones
    #subtask2(file_name)
    #uncomment for subtask 3 where I use the red pass filter
    subtask3(file_name)


