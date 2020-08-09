#!/usr/bin/env python3
"""
Documentation

See also https://www.python-boilerplate.com/flask
"""
import os
import glob
import argparse
import json
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from helper import *
import concurrent.futures
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def create_app(config=None):
    app = Flask(__name__)
    app.config.update(dict(DEBUG=True))
    app.config.update(config or {})

    # Setup cors headers to allow all domains
    CORS(app)

    # Definition of the routes.
    @app.route("/filter")
    def filter():
        try:
            # Read CSV file
            df = pd.read_csv(app.config['input_csv_file'])
            # Use only filename and gender columns and filter out null or other values for gender
            new_df = df[["filename", "gender"]]
            new_df = new_df[np.logical_or(new_df['gender'] == 'female', new_df['gender'] == 'male')]
            # Balance the values of the dataframe to avoid overfitting
            balanced = new_df.groupby('gender')
            balanced = pd.DataFrame(balanced.apply(lambda x: x.sample(balanced.size().min()).reset_index(drop=True)))

            # Save the new file
            if not os.path.isdir("results"):
                os.mkdir("results")
            balanced.to_csv("results/balanced_filtered_audio.csv", index=False)

            balanced['filename'] = balanced['filename'].str.replace(r'.mp3$', '.npy')
            balanced['filename'] = balanced['filename'].map(lambda x: "results/features/"+os.path.basename(x))
            balanced.to_csv("results/balanced_filtered_features.csv", index=False)

            return jsonify({"results": "success"})
        except Exception as e:
            return jsonify({"result": "failure", "exception":str(e)})

    @app.route("/generate")
    def generate():
        try:
            df = pd.read_csv("results/balanced_filtered_audio.csv")
            audio_files = [app.config['input_audio_path'] + filename for filename in df['filename'].values.tolist()]

            if not os.path.isdir("results/features/"):
                os.mkdir("results/features/")

            with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
                tile = {executor.submit(extract_features, audio_file,"results/features/"): audio_file for audio_file in audio_files}

            return jsonify({"result": "success"})
        except Exception as e:
            return jsonify({"result": "failure", "exception":str(e)})

    @app.route("/train/<network_type>")
    def train(network_type):
        try:
            # load the dataset
            X, y = load_data()
            # split the data into training, validation and testing sets
            data = split_data(X, y, test_size=0.1, valid_size=0.1)
            accuracy_result = 0
            if(network_type == "feedforward"):
                # construct the model
                model = create_model()

                # train the model
                batch_size = 64
                epochs = 100
                model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
                        callbacks=[EarlyStopping(mode="min", patience=10, restore_best_weights=True)])

                # evaluate the model accuracy using the testing set
                loss, accuracy = model.evaluate(data["X_train"], data["y_train"], verbose=0)
                print("Accuracy on training set: {:.3f}".format(accuracy))
                loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
                print("Accuracy on test set: {:.3f}".format(accuracy))
                accuracy_result = accuracy

            elif network_type == "randomforest":
                forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(data["X_train"], data["y_train"])
                print("Accuracy on training set: {:.3f}".format(forest.score(data["X_train"], data["y_train"])))
                accuracy_result = forest.score(data["X_test"], data["y_test"])
                print("Accuracy on test set: {:.3f}".format(accuracy_result))
            
            elif network_type == "decisiontree":
                tree = DecisionTreeClassifier(random_state=0).fit(data["X_train"], data["y_train"])
                print("Decision Tree")
                print("Accuracy on training set: {:.3f}".format(tree.score(data["X_train"], data["y_train"])))
                accuracy_result = tree.score(data["X_test"], data["y_test"])
                print("Accuracy on test set: {:.3f}".format(accuracy_result))
            elif network_type == "gradientboosting":
                gbrt = GradientBoostingClassifier(random_state=0).fit(data["X_train"], data["y_train"])
                print("Gradient Boosting")
                print("Accuracy on training set: {:.3f}".format(gbrt.score(data["X_train"], data["y_train"])))
                accuracy_result = gbrt.score(data["X_test"], data["y_test"])
                print("Accuracy on test set: {:.3f}".format(accuracy_result))
            elif network_type == "mlp":
                mlp = MLPClassifier(random_state=0).fit(data["X_train"], data["y_train"])
                print("Multilayer Perceptron")
                print("Accuracy on training set: {:.3f}".format(mlp.score(data["X_train"], data["y_train"])))
                accuracy_result = mlp.score(data["X_test"], data["y_test"])
                print("Accuracy on test set: {:.3f}".format(accuracy_result))

            return jsonify({"result": "success","accuracy":accuracy_result})
        except Exception as e:
            return jsonify({"result": "failure", "exception":str(e)})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default="8000")

    args = parser.parse_args()
    port = int(args.port)

    config = json.load(open("config.json"))
    app = create_app(config)
    app.run(host="0.0.0.0", port=port)