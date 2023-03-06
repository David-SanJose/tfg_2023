# Importing the OpenCV library
import cv2
# Reading the image using imread() function
image = cv2.imread('chad.png')

output = image.copy()
  
# Using the rectangle() function to create a rectangle.
rectangle = cv2.rectangle(output, (0, 0), 
                          (400, 400), (255, 0, 0), 2)
cv2.imshow("hila", rectangle)

cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()