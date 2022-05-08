import cv2
import numpy as np
from numba import jit
import time

start_time = time.time()
# Размер квадратной маски
N = 4
def oppening(image):    # морфологическая трансформация
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.dilate(cv2.erode(image, kernel), kernel)
    return res
def square(image):
    side = 600
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    return image[int((height-side)/2):int((height+side)/2), int((width-side)/2):int((width+side)/2)]
def edges():    #back, vert, hor, diag1, diag2
    list =[]
    vert = hor = diag2 = np.zeros((N, N))
    vert[:, int(N / 2) - 1:int(N / 2) + 1] = np.full((N, 2), 255)
    hor[int(N / 2) - 1:int(N / 2) + 1, :] = np.full((2, N), 255)
    for i in range(N):
        diag2[i, N - 1 - i] = 255
    list.append(np.zeros((N, N)))
    list.append(vert)
    list.append(hor)
    list.append(np.eye(N)*255)
    list.append(diag2)
    return list
def proj_vertical(image, N):
    measure = (N/2)**2
    # считаем суммарную яркость на разных областях постоянной яркости образца
    bright1 = np.sum(image[:, 0:int(N/2)])
    bright2 = np.sum(image) - bright1
    # проекция на изображение
    proj = np.full((N, N), bright2 / measure)
    proj[:, 0:int(N/2)] = np.full((N, int(N/2)), bright1/measure)
    return proj
def proj_horizontal(image, N):
    measure = (N / 2) ** 2
    # считаем суммарную яркость на разных областях постоянной яркости образца
    bright1 = np.sum(image[0:int(N / 2), :])
    bright2 = np.sum(image) - bright1
    # проекция изображение
    proj = np.full((N, N), bright2 / measure)
    proj[0:int(N / 2), :] = np.full((int(N / 2), N), bright1 / measure)
    return proj
def proj_diagonal1(image, N):
    measure1 = N*(N+1)/2
    measure2 = N*(N-1)/2
    bright1 = 0
    # считаем суммарную яркость на разных областях постоянной яркости образца
    for i in range(N):
        for j in range(i+1):
            bright1 += image[i, j]
    bright2 = np.sum(image) - bright1
    # проекция на изображение
    proj = np.full((N, N), bright2 / measure2)
    for i in range(N):
        for j in range(i + 1):
            proj[i, j] = bright1/measure1
    return proj
def proj_diagonal2(image, N):
    measure1 = N * (N + 1) / 2
    measure2 = N * (N - 1) / 2
    bright1 = 0
    # считаем суммарную яркость на разных областях постоянной яркости образца
    for i in range(N):
        for j in range(N - i):
            bright1 += image[i, j]
    bright2 = np.sum(image) - bright1
    # проекция на изображение
    proj = np.full((N, N), bright2 / measure2)
    for i in range(N):
        for j in range(N - i):
            proj[j, i] = bright1 / measure1
    return proj
def proj_backlight(image,N):
    measure = N**2
    bright = np.sum(image)
    proj = np.full((N, N), bright/measure)
    return proj
def all_proj(frag, N): # возвращает лист всех проекций
    return (proj_backlight(frag, N), proj_vertical(frag, N), proj_horizontal(frag, N), proj_diagonal1(frag, N), proj_diagonal2(frag, N))
@jit(fastmath = True)
def distance(image, proj): #возвращает расстояние м/у изображением и его проекцие
    temp = np.square(image - proj)
    return np.sum(temp)
def sort(image, projs): # возвращает индекс наиболее подходящей проеции
    list = []
    for i in range(5):
        list.append(distance(image, projs[i]))
    return list.index(min(list))
def morf_edges(image, N):
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    res = image
    list = edges()
    # проходимся маской по изображению, и выделяем фрагменты 6x6 пикселей
    for i in range(0, height // N):
        for j in range(0, width // N):
            frag = image[i*N: i*N + N, j*N: j*N + N]
            projs = all_proj(frag, N) #cоздаем массив проекций фрагмента
            for line in range(0, height // N):
                for col in range(0, width // N):
                    # формируется вторичное изображение
                    res[i*N: i*N + N, j*N: j*N + N] = list[sort(frag, projs)]
    return res

# считываем изображение и перевод в оттенки серого
image = cv2.cvtColor(cv2.imread("Photo\Vinni.jpg"), cv2.COLOR_BGR2GRAY)
# сохраняем обработанное изображение
cv2.imwrite('Edges\ex_11.jpg', morf_edges(image, N))

#вывод на экран
cv2.imshow("Image", cv2.imread('Photo\ex_1.jpg'))
cv2.imshow("result", cv2.imread('Edges\ex_1.jpg'))

print("--- %s seconds ---" % (time.time() - start_time))

cv2.waitKey(0)
cv2.destroyAllWindows()