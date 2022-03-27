import cv2
import numpy as np

    ### reading image and displaying it
#
# img = cv2.imread("python_face_changing.jpg")
# cv2.imshow("Picture", img)
#
# cv2.waitKey(0)

    ### reading and displaying video

# cap = cv2.VideoCapture("sample_1920x1080.mp4")
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


    ### reading and displaying webcam

# cap = cv2.VideoCapture(0)
# # cap.set(3, 640)
# # cap.set(4, 480)
# cap.set(10, 100)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Video", img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

    ### Editing image

img = cv2.imread("python_face_changing.jpg")
kernel = np.ones((5,5), np.uint8)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (15,15), 0)
# Finding edges
img_canny = cv2.Canny(img, 100, 100)
img_dilation = cv2.dilate(img_canny, kernel, iterations = 1)
img_eroded = cv2.erode(img_dilation, kernel, iterations = 1)


cv2.imshow("Gray Image", img_gray)
cv2.imshow("Normal Image", img)
cv2.imshow("Blur Image", img_blur)
cv2.imshow("Canny Image", img_canny)
cv2.imshow("Image Dilation", img_dilation)
cv2.imshow("Image Erodion", img_eroded)
cv2.waitKey(0)

    ### Resizing image

# img = cv2.imread("python_face_changing.jpg")
# img_resize = cv2.resize(img, (715, 400))
# img_cropped = img[300:400, 400:1000]
#
#
# print(img.shape)
# print(img_resize.shape)
#
# cv2.imshow("Image", img)
# cv2.imshow("Image Resized", img_resize)
# cv2.imshow("Image Cropped", img_cropped)
#
#
# cv2.waitKey(0)


    ### Making custom image

# img = np.zeros((512, 512, 3), np.uint8)
# # img[:] = 255, 255, 0
#
# cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
# cv2.rectangle(img, (0,0), (250, 350), (0, 0, 255), 2)
# cv2.circle(img, (400, 50), 30, (255, 255, 0), 5)
# cv2.putText(img, "OPENCV DISPLAYING TEXT", (50, 400), cv2.FONT_ITALIC, 1, (0, 150, 0), 1)
#
#
# cv2.imshow("Image", img)
#
# cv2.waitKey(0)


    ### WARP PERSPECTIVE


# img = cv2.imread("cards.png")
#
# width, height = 250, 350
#
# pts1 = np.float32([[76,157], [206, 133], [108, 350], [256, 320]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))
#
#
#
# cv2.imshow("Image", img)
# cv2.imshow("Warp perspective", img_output)
#
# cv2.waitKey(0)


    ### JOINING IMAGES

# img = cv2.imread("python_face_changing.jpg")
#
# img_hor = np.hstack((img, img))
# img_vert = np.vstack((img, img))
#
# cv2.imshow("Horizontal", img_hor)
# cv2.imshow("Vertical", img_vert)
#
# cv2.waitKey(0)


    ### COLOR DETECTION

# img = cv2.imread("python_face_changing.jpg")
#
# image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#
# cv2.imshow("Original", img)
# cv2.imshow("HSV", image_HSV)
#
# cv2.waitKey(0)


    ### TEST AREA 1

# img = cv2.imread("jaca.jpeg")
# # img_resize = cv2.resize(img, (715, 400))
# # width, height = 250, 350
#
#
# width, height = 300, 400
# pts1 = np.float32([[286,916], [805, 511], [680, 1560], [1149, 1262]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))
#
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
# image_sharp_output = cv2.filter2D(src=img_output, ddepth=-1, kernel=kernel)
#
# img_resize = cv2.resize(image_sharp_output, (715, 800))
# img_resize_sharp = image_sharp_output = cv2.filter2D(src=img_resize, ddepth=-1, kernel=kernel)
# cv2.imshow('Image Sharp', image_sharp)
# cv2.imshow("Card Sharp", image_sharp_output)
# cv2.imshow("Image resize", img_resize_sharp)
# img_canny = cv2.Canny(img_output, 100, 100)
# cv2.imshow("Canny", img_canny)
# # cv2.imshow("Image", img)
# cv2.imshow("Warp perspective", img_output)

cv2.waitKey(0)

    ### TEST AREA 2

# img = cv2.imread("jaca.jpeg")
#
# width, height = 700, 900
#
# pts1 = np.float32([[286,916], [805, 511], [680, 1560], [1149, 1262]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))
# img_canny = cv2.Canny(img_output, 100, 100)
#
# # cv2.imshow("Image", img)
# cv2.imshow("Warp perspective", img_output)
# cv2.imshow("Image canny", img_canny)
#
#
# cv2.waitKey(0)