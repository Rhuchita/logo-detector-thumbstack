from flask import Flask, flash, render_template, request, redirect, url_for
from sklearn.neighbors import KNeighborsClassifier

# from werkzeug.utils import secure_filename
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os

app = Flask(__name__)
# Get Paths
trainingPath = "Dataset\Logos"
testPath = "Dataset\Testing"

UPLOAD_FOLDER = "Dataset/uploads/"
app.secret_key = "None"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image


# def train_set(trainingPath):


# Check Test Images for Model
def predict_img(impath):
    for imagePath in glob.glob(trainingPath + "/*/*.*"):
        # get label from folder name
        label = imagePath.split("\\")[-2]

        image = cv.imread(imagePath)
        try:
            # RGB to Gray
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Calculate Low and Up value to extract Edges
            md = np.median(gray)
            sigma = 0.35
            low = int(max(0, (1.0 - sigma) * md))
            up = int(min(255, (1.0 + sigma) * md))
            # Create Edged Image from Gray Scale
            edged = cv.Canny(gray, low, up)

            # extract only shape in image
            (x, y, w, h) = cv.boundingRect(edged)
            logo = gray[y : y + h, x : x + w]
            logo = cv.resize(logo, (200, 100))

            # Calculate histogram
            hist = feature.hog(
                logo,
                orientations=9,
                pixels_per_cell=(10, 10),
                cells_per_block=(2, 2),
                transform_sqrt=True,
                block_norm="L1",
            )
            # Add value into Lists
            hists.append(hist)
            labels.append(label)
        except cv.error:
            # If Image couldn't be Read
            print("Training Image couldn't be read")

    # Create model as Nearest Neighbors Classifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(hists, labels)

    image = cv.imread(impath)
    try:
        # Convert RGB to Gray and Resize
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        logo = cv.resize(gray, (200, 100))
        # Calculate Histogram of Test Image
        hist = feature.hog(
            logo,
            orientations=9,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L1",
        )
        # Predict in model
        predict = model.predict(hist.reshape(1, -1))[0]
        # Make pictures default Height
        height, width = image.shape[:2]
        reWidth = int((300 / height) * width)
        image = cv.resize(image, (reWidth, 300))

    except cv.error:
        # If Image couldn't be Read
        print(impath)
        print("Test Image couldn't be read")
        return "Unable to predict"
    return predict.title()


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["image"]

        img_path = "Dataset/uploads/" + img.filename
        img.save(img_path)
        # img.save(os.path.join(app.config["UPLOAD_FOLDER"], img.filename))
        p = predict_img(img_path)
        print(p)
        return render_template("index.html", prediction=p, img_path=img.filename)
    return render_template("index.html")


@app.route("/submit/<filename>")
def display_image(filename=""):
    from flask import send_from_directory

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if "__name__" == "main":
    app.run(debug=True)
