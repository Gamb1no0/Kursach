import cv2

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("Photo\Ksusha.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edgeX= cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
edgeY= cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
edge= cv2.Sobel(image,cv2.CV_64F,1,1,ksize=3)
edges = cv2.Canny(image,50,00)
viewImage(image, "kotik")
viewImage(edge, "Sobely")
viewImage(edges, "Sobely")


cv2.waitKey(0)
cv2.destroyAllWindows()