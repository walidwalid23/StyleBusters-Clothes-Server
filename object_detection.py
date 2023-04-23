import os
import shutil
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename
from urllib.request import urlopen

# OBECT DETECTION FUNCTION


def obejectDetection(sentClassName, imageFile=None, imageURL=None):
    # Initialize Obeject Detector
    yolo_model = YOLO("object_detector/yolov8.pt")
    # Load The Image
    if imageFile != None:
        # for input image
        uploadedImage = Image.open(imageFile.stream)
        uploadedImageName = secure_filename(imageFile.filename)
        croppedImageName = "."+uploadedImageName.split(".")[-1]+".jpg"
    else:
        # for retrieved images
        uploadedImage = Image.open(urlopen(imageURL)).convert('RGB')
        uploadedImageName = imageURL.split("/")[-1]
        croppedImageName = "image0.jpg"

    print("Uploaded Image Name: "+uploadedImageName)
    # Note:results is an array and each element of this array is a single frame(image) detection
    # so we only need results[0] since we are dealing with a single image
    # not a video or various images at once
    results = yolo_model.predict(uploadedImage, save_crop=True,
                                 conf=0.5, project="runs", name=uploadedImageName, exist_ok=True)

    classesIndices = results[0].boxes.cls
    classesCount = len(classesIndices)
    # make conditions in case class count is more than 1
    print("Classes Count is " + str(classesCount))
    # iterate over all classes to send them to the user to choose from
    classNames = []
    for classIndex in classesIndices:
        className = results[0].names[int(classIndex)]
        classNames.append(className)

    # if no classes were detected
    if classesCount == 0:
        return {"noClothes": True}
    else:
        # IF SENTCLASSNAME=NULL THIS MEANS THAT THIS IS THE FIRST REQUEST THE USER MAKE THEREFORE WE SHOULD MAKE HIM CHOOSE
        if len(classNames) > 1 and sentClassName == None:
            # remove the directory after loading the image"
            shutil.rmtree('runs/'+uploadedImageName)
            # return the classes to choose from and the bounding boxes to display on frontend
            return {"multiClass": True,
                    "classes": classNames,
                    "boundingBoxes": results[0].boxes.xywh.numpy()}

        # IF THE USER MADE A REQUEST BEFORE AND ALREADY SELECTED A CLASS FIND THE CLASS HE SELECTED

        elif len(classNames) > 1 and sentClassName != None:
            for className in classNames:
                if className == sentClassName:
                    predictedCroppedImagePath = "runs/"+uploadedImageName + \
                        "/crops/" + className+"/" + croppedImageName
                    croppedImage = Image.open(
                        predictedCroppedImagePath).convert('RGB')
                    # remove the directory after loading the image"
                    shutil.rmtree('runs/'+uploadedImageName)
                    # return the the class name and the cropped image
                    # replace _ with space to get search results
                    return {"className": sentClassName.replace("_", " "),
                            "croppedImage": croppedImage}
                else:
                    # IF THE CLASSNAME SENT WAS NOT FOUND RETURN NONE (THIS COULD HAPPEN FROM RETRIEVED IMAGES NOT FROM USER SIDE)
                    return {"croppedImage": None}

        # IF THIS IS THE FIRST REQUEST BUT THERE IS ONLY ONE DETECTED CLASS
        else:
            # load cropped class image (AT THIS POINT YOU SHOULD KNOW WHICH CLASS THE USER WANT THEN ONLY LOAD ITS CROP OR IF ONLY 1 CLASS)
            predictedCroppedImagePath = "runs/"+uploadedImageName + \
                "/crops/" + classNames[0]+"/" + croppedImageName
            croppedImage = Image.open(predictedCroppedImagePath).convert('RGB')
            # remove the directory after loading the image"
            shutil.rmtree('runs/'+uploadedImageName)
            # return the the class name and the cropped image
            return {"className": classNames[0].replace("_", " "),
                    "croppedImage": croppedImage}
