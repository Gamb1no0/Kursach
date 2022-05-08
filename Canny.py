import cv2
import numpy as np

def square(image):
    side = 600
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    return image[int((height-side)/2):int((height+side)/2), int((width-side)/2):int((width+side)/2)]

image = cv2.cvtColor(cv2.imread("Photo\ex_1.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_1.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_2.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_2.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_3.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_3.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_4.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_4.jpg', cv2.Canny(image,50,100))
image = cv2.cvtColor(cv2.imread("Photo\ex_6.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_6.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_7.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_7.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_9.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_9.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_12.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_12.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\ex_13.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_13.jpg', cv2.Canny(image,100,200))
image = cv2.cvtColor(cv2.imread("Photo\Vinni.jpg"), cv2.COLOR_BGR2GRAY)
cv2.imwrite('Canny\ex_14.jpg', cv2.Canny(image,100,200))

cv2.waitKey(0)
cv2.destroyAllWindows()