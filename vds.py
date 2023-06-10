import cv2
import numpy as np

# Variable initialization
car_count = []
mask = None
prev_frame = None
prev_comp_frame = None
roi = None
# Change Cars.mp4 with your traffic road video
vid = cv2.VideoCapture("cars.mp4")
# It's an approximation of how many blocks will be counted as a car, should be adjusted case by case
block_to_car_ratio = 35
# Pointer size when drawing the mask
arrow = 60


def draw_mask(event, x, y, flag, param):
    '''
    @function draw_mask
    @brief This function allows the user to draw circles within a window with the mouse.
    You can leave the mouse clicked or just click once on desired areas.

    @note function shared by the professor
    '''
    global ix2, iy2, drawing, arrow
    # left click pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # pointer position
        ix2, iy2 = x, y
        # mouse movement
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # draws circle in x,y pos
            cv2.circle(mask, (x, y), arrow, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        # button not pressed
        drawing = False
        cv2.circle(mask, (x, y), arrow, (255, 255, 255), -1)


def frame_comparison(actual_frame, previous_frame, image_mask, previous_comparison_frame):
    '''
    @function frame_comparison
    @brief This function makes multiple comparisons between frames and image filtering. It applies
        a threshold to the comparison to simplify the image comparison as black and white instead of
        grayscale. After that it applies an image dilation to remove noise from the image, caused
        by shadows, reflections or small movements. Then a dilation is applied to increase the size
        of the bigger blocks that represent bigger objects. Finally a Gaussian Blur is applies to
        smooth the transition between edges.

    @param actual_frame - current frame from the video
    @param previous_frame - previous frame from the video
    @param image_mask - mask created to designate an area of interest for the frame comparisons.
        It is useful to leave out areas from the video that could cause unnecessary noise.
    @param previous_comparison_frame - previous image comparison to find differences between frames.

    @return comparison - the frame comparison after processed by different filtering techniques
    @return previous_comparison_frame - previous image comparison to find differences between frames.
    @return previous_frame - previous frame from the video
    '''
    # Initialization of erosion and dilation kernels. Int values can be changed to adjust to specific applications.
    kernel_erode = np.ones((4, 4), np.uint8)
    kernel_dilate = np.ones((20, 20), np.uint8)
    # frame variable assignment and comparison between actual and previous frame
    if previous_frame is None:
        previous_frame = actual_frame
        # frame comparison within mask region
        previous_comparison_frame = cv2.bitwise_and(actual_frame, previous_frame, mask=image_mask)
        comparison = None
    else:
        comparison_frame = cv2.bitwise_and(actual_frame, previous_frame, mask=image_mask)
        comparison = cv2.absdiff(previous_comparison_frame, comparison_frame)
        previous_comparison_frame = comparison_frame
        previous_frame = actual_frame
        # Used to leave the processed image in black and white
        _, comparison = cv2.threshold(comparison, 30, 255, cv2.THRESH_BINARY)
        # Erode is used to erase noise from the image
        comparison = cv2.morphologyEx(comparison, cv2.MORPH_ERODE, kernel_erode)
        # Dilate is to increase the size of the remaining image regions
        comparison = cv2.morphologyEx(comparison, cv2.MORPH_DILATE, kernel_dilate)
        comparison = cv2.GaussianBlur(comparison, (11, 11), 5)
    return comparison, previous_comparison_frame, previous_frame


def car_contour_counter(compared_images, roi_coords, car_counter_list):
    '''
    @name car_contour_counter
    @brief This function uses OpenCV's function findContours to create contours on image regions, which
        then are related by hierarchy to tell parent from child blocks. These blocks are then analyzed
        in the region of interes (ROI) to be counted as cars if the block size is between some x & y
        parameters

    @param compared_images - processed image comparison between frames
    @param roi_coords - Coordinates from an assigned Region of Interest
    @param car_counter_list - list that keeps track of the blocks detected on the image

    @return contours generation of contours within certain image regions related to objects
    @return car_counter_list list of objects detected, which needs to be normalized case by case
    '''
    contours, hierarchy = cv2.findContours(compared_images, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for car in contours:
        # boundingRect delimits contour approximations
        x, y, w, h = cv2.boundingRect(car)
        if (x <= (roi_coords[0] + roi_coords[2])) & (y >= (roi_coords[1] + roi_coords[3])):
            car_counter_list.append(car)
    return contours, car_counter_list


while vid.isOpened():
    _, image = vid.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Selects region of interest if one hasn't been picked yet
    if roi is None:
        roi = cv2.selectROI(image)
    # Selects a mask if one hasn't been selected yet
    elif mask is None:
        drawing = False
        alpha = 0.7
        mask = np.zeros(gray.shape, np.uint8)
        cv2.namedWindow('Create Mask')
        # Mouse events inside window within cv
        cv2.setMouseCallback('Create Mask', draw_mask)
        # alpha color over the image
        overlay = gray.copy()
        # creates a copy of the image
        output = gray.copy()
        # Mask creation - Separate window to draw mask region
        while True:
            cv2.addWeighted(overlay, alpha, mask, 1 - alpha, 0, output)
            cv2.imshow('Create Mask', output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Exit the mask selection loop when 'q' is pressed
                break
        cv2.destroyAllWindows()
    else:
        # Draws the ROI rectangle in the main colored image
        cv2.rectangle(image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 255), 1)
        cv2.rectangle(gray, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 255), 1)
        # initialization comparison, verifies frame differences and removes noise from the comparison image
        comparison, prev_comp_frame, prev_frame = frame_comparison(gray, prev_frame, mask, prev_comp_frame)
        # Runs function to count objects on ROI and returns the car_counter_list
        contours, car_count = car_contour_counter(comparison, roi, car_count)
        print("Number of cars:", len(car_count))

        cv2.imshow('Video', image)
        cv2.imshow('Comparison', comparison)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' is pressed
            break

vid.release()
cv2.destroyAllWindows()
