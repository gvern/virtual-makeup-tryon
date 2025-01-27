import cv2

def load_image(path):
    image = cv2.imread(path)
    return image

def save_image(path, image):
    cv2.imwrite(path, image)
