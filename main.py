import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
import os
from PIL import Image
from urllib.request import urlopen
import json
import requests
from dotenv import load_dotenv
from object_detection import obejectDetection

global embed

load_dotenv()


# allowed clothes images extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# preparing the server
app = Flask(__name__)


# Instantiate a VGG model with our saved weights
vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)  # B
# load the model using your available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.classifier[-1] = nn.Linear(4096, 46)
vgg_model.load_state_dict(torch.load('feature_extractor/vgg11_4_deepfashion1.pt',
                                     map_location=torch.device(device)))
vgg_model.eval()


@app.route('/', methods=['GET'])
def root():
    return jsonify(
        {"you are at root": "This is the root of stylebusters clothes app"})


@app.route('/get-similar-clothes', methods=['POST'])
def getSimilarClothes():
    print("in route")
    image = request.files['image']
    sort = request.form['sort']
    page = request.form['page']
    sentClassName = request.form['className'] if 'className' in request.form else None

    if image and allowed_file(image.filename):
        session = requests.Session()
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"errorMessage": "no selected image"})
        # call the object detection function
        objectDetectionResult = obejectDetection(
            sentClassName, imageFile=image)

        if objectDetectionResult.get("noClothes") is not None:
            return jsonify({"errorMessage": "Your Image Doesn't Contain Visible Clothes"})

        elif objectDetectionResult.get("multiClass") is not None:
            return jsonify({"successMessage": "Image Contains More Than One Class",
                            "classes": objectDetectionResult["classes"],
                            "boundingBoxes": objectDetectionResult["boundingBoxes"]})
        else:
            className = objectDetectionResult["className"]
            croppedImage = objectDetectionResult["croppedImage"]
            print("new name: "+className)

            # getting feature vector of the cropped clothes image
            _transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    # standardization of pixel colors
                    transforms.Normalize([0.485, 0.456, 0.406], [
                        0.229, 0.224, 0.225])
                ])

            input_image_matrix = np.asarray(
                np.expand_dims(_transforms(croppedImage), 0))
            input_image_vector = vgg_model.features(
                torch.tensor(input_image_matrix)).mean((2, 3))

            # getting feature vectors of the retrieved clothes images
            URL = "https://unofficial-shein.p.rapidapi.com/products/search?keywords=" + \
                className+"&language=en&country=EGY&sort="+sort+"&page="+page+"&limit=300"

            headers = {
                'user-agent': 'Mozilla/5.0',
                "X-RapidAPI-Key": "a6c2c31630msh991be42e1cce8e5p1ec51ejsn2ad953cf202e",
                "X-RapidAPI-Host": "unofficial-shein.p.rapidapi.com"
            }

            response = session.get(URL, headers=headers)
            jsonResponse = json.loads(response.text)
            # print(jsonResponse)
            products = jsonResponse["info"]["products"]
            results = []

            print(products)

            for product in products:
                clothesImageUrl = product["goods_img"]
                # extract features from each retrieved clothes image
                # call the object detection function
                retrievedObjectDetectionResult = obejectDetection(
                    className.replace(" ", "_"), imageURL=clothesImageUrl)
                # if there are clothes detections pass
                if objectDetectionResult.get("noClothes") is not None:
                    print("no clothes detected")
                    pass
                elif "croppedImage" in retrievedObjectDetectionResult:
                    print("searching")
                    retrievedCroppedImage = retrievedObjectDetectionResult["croppedImage"]
                    # if atleast one the classes detected in retrieved images matches the class detected in input image or selected by user
                    if retrievedCroppedImage != None:

                        retrieved_image_matrix = np.asarray(
                            np.expand_dims(_transforms(retrievedCroppedImage), 0))
                        retrieved_image_vector = vgg_model.features(
                            torch.tensor(retrieved_image_matrix)).mean((2, 3))

                        # get the cosine similarity
                        cosine_similarity = torch.cosine_similarity(
                            input_image_vector, retrieved_image_vector)
                        string_cosine_similarity = str(
                            cosine_similarity)[8:12]+" %"
                        # print("Cosine Similarity of The Main Image and Image:" +
                        #      product["goods_img"] + " is: " + string_cosine_similarity+" ")
                        # URL CONSISTS OF DOMA/goods_url_name (with spaces replaced with -) + -p- + goods_id + -cat- + cat_id + .html
                        if cosine_similarity > 0.7:
                            print("found a match")
                            results.append({"imageURL": product["goods_img"],
                                            "accuracy": cosine_similarity})

            for result in results:
                print("result: "+str(result["imageURL"]) +
                      " Accuracy: "+str(result["accuracy"]))
            return jsonify({"successMessage": "Done Searching"})

    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})


# check if this file is being excuted from itself or being excuted by being imported as a module
if __name__ == "__main__":
    from waitress import serve
    print("server is running at port "+str(os.getenv("PORT")))
    serve(app, port=os.getenv("PORT"))
