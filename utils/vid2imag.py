import cv2
vidcap = cv2.VideoCapture('/home/charvi/workspace/iiit-hyderabad/civil-inspection/uvrsabi_code/UVRSABI-Code/Flight_loger/CBRI_Data/Physics_Roof/Raw_Videos/Distance_between_buildings/DJI_0372.MP4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(str(count)+".png", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)