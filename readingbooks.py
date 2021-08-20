##### for tessseract to run first do ###########
# sudo apt install tesseract & pip install pytesseract scikit-image and 
# recommended to update imutils with pip install -U imutils
#### to run, python readingbooks.py -s <source>
## what can be source? its either path to image or use webcam ( to use webcam pass 0 or 1 if you 
# have an external webcam)
#  to morph the image during image processing(only applicable for images )use -m True (  its false
# by default)

############## import necessary libraries#######

import cv2
import numpy as np
import argparse 
import imutils
from skimage.filters import threshold_local
from imutils.perspective import four_point_transform
import pytesseract


ap = argparse.ArgumentParser()
ap.add_argument('-s','--source',required= True)

ap.add_argument('-m','--morph',required=False,default=False,
                help='Either to morph the image or not')
args = vars(ap.parse_args())
 
############# using pytesseract to get output from image after thresholding it
# if just thresholding doesnt work , pass parameter sk=True for more precision###############
def gettext(img):
    img = thresholding(img,sk=True)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    text = pytesseract.image_to_string(img)
    
    return text


##########to rearrange the 4 corners of contours so that smallest one comes first and
# so on , as to make a rectangle points in order##################
def rearrangement(pts):
    npts = np.zeros_like(pts)
    pts = pts.reshape((4,2))
    sum = pts.sum(1)
   
    npts[0]= pts[np.argmin(sum)]
    npts[3]= pts[np.argmax(sum)]
    
    diff = np.diff(pts,axis=1)
    npts[1]= pts[np.argmin(diff)]
    npts[2]= pts[np.argmax(diff)]
    return npts
######## morphing filter , use only in images as its quite cpu consuming#######
def morphing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=1)
    return closing
######## self explanatory , LOLx
def preProcessing(img):
    imgBlur = cv2.GaussianBlur(img,(5,5),0)
    imgCanny = cv2.Canny(imgBlur,100,200)
    return imgCanny
########### make image more noise free if there is any, you may want to add it in the result
# if you dont want to see any noise but this may sometimes denoise the letters so
# i only used it for first preprocessing process##############
def denoiser(img):
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    return dst
##basic thresholding 
def thresholding(img,sk=False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # thres = cv2.adaptiveThreshold(gray,\
    #                         255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                         cv2.THRESH_BINARY,7,1.3)
    if sk:
        T = threshold_local(gray, 15, offset = 10, method = "gaussian")
        thres = (gray > T).astype("uint8") * 255
        return thres
    _,thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thres
#### finding contours and returning the biggest rectangle among them
def findcontours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(finalimage, contours, -1, (255, 155, 0), 3)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            
            if len(approx) == 4:
                mainbox = approx
                # cv2.drawContours(finalimage, mainbox, -1, (0, 0, 255), 3)
               
                return mainbox 
    return np.array([0,0])
##### run the application from webcam or camera , ps no video yet 
def fromvideo(source):
    cap = cv2.VideoCapture(source)

    while True:
        _,img = cap.read()
        showingmaterial= img.copy()
        ratio = img.shape[1]/400
        img = imutils.resize(image=img,width=400)
        finalimage = img.copy()

        # dnt = denoiser(img)
        thres = thresholding(img)
        if args['morph']:
            morphed = morphing(thres)
            canny = preProcessing(morphed)
        else:
            canny = preProcessing(thres)
        pts = findcontours(canny)
        if pts.sum() != 0:
            
            result = four_point_transform(showingmaterial, pts.reshape(4, 2)* ratio )
            # res = thresholding(result)
            # filename =writefile(result,T=True)
            text = gettext(img)
            print(text)
            cv2.imshow('result',result)
   

        cv2.imshow('camera',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#### run the application from image
def fromImg(path):
    global finalimage
    img = cv2.imread(path)
    showingmaterial= img.copy()
    ratio = img.shape[1]/300
    img = imutils.resize(image=img,width=300)
   
    finalimage = img.copy()
    dnt = denoiser(img)
    thres = thresholding(dnt)

    if args['morph']:
        morphed = morphing(thres)
        canny = preProcessing(morphed)
    else:
        canny = preProcessing(thres)

    pts = findcontours(canny)
    # cv2.imshow('cany image',canny)
    cv2.imshow('Original image',img)
    # cv2.imshow('morphed',morphed)
    # cv2.imshow('thres',thres)
  
    if pts.sum() != 0:
       
        result = four_point_transform(showingmaterial, pts.reshape(4, 2)* ratio )
        res = thresholding(result,sk=True)

        cv2.imshow('result',res)
    cv2.waitKey(0)
 ###### main , if int is given then use video else use the source 
def main():
    source = args['source']
    try:
        int(source)
        fromvideo(int(source))
    except:
        fromImg(source)


if __name__ == '__main__':
    main()
## bad documentation but hum aise hi hain lol