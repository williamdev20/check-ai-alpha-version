import cv2

def processing_image():
    img = cv2.imread("./images/cartaz-dengue.jpeg")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: ignore
    #blur = cv2.medianBlur(gray_image, 3)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    contrast = clahe.apply(gray_image)

    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    resized_image = cv2.resize(contrast, dsize=None, dst=None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) # type: ignore
    

    cv2.imshow("Imagem redimensionada", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return thresh


if __name__ == "__main__":
    processing_image()
