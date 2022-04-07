from poplib import POP3_PORT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import numpy as np
import pandas as pd
from os.path import exists
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import gradio as gr

objects = {
    0: "Bowtie",
    1: "Broom",
    2: "Crown",
    3: "EiffelTower",
    4: "HotAirBalloon",
    5: "HousePlant",
    6: "Bed",
    7: "Cat",
    8: "Couch",
    9: "Dog",
    10: "Hand",
    11: "Hat",
    12: "Tractor"
}

data = pd.DataFrame()

# Load data from all npy files
for object in objects:

    # Load the numpy file
    object_data = None
    if exists(f"./data/{objects[object]}.npy"):
        object_data = np.load(f"./data/{objects[object]}.npy")
    else:
        object_data = np.load(
            f"./DoodleClassifierModel/data/{objects[object]}.npy")

    # Add labels to data
    temp = pd.DataFrame(object_data)
    temp["Label"] = object

    # Append object data to main dataframe
    data = pd.concat([data, temp], ignore_index=True)


# Train test validation split
x_train, x_test, y_train, y_test = train_test_split(
    data.loc[:, data.columns != "Label"], data["Label"], test_size=0.33, random_state=69)

model = RFC(n_estimators=100, max_depth=None, random_state=420)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

def show_confusion_matrix():
    # Create labels
    labels = list(objects.keys())

    # Initialize confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=predictions, labels=labels)
    print(cm)

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')

    fig.colorbar(cax)

    # "If you have more than a few categories, Matplotlib decides to label the axes incorrectly - you have to force it to label every cell." - https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([''] + list(objects.values()))
    ax.set_yticklabels([''] + list(objects.values()))
    ax.tick_params(labelrotation=45)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Draw figure
    plt.show()


# show_confusion_matrix()

def make_prediction(pred):
    """Makes a prediction using the model, takes in a pandas series, aka a single row from a pandas df, or an array of the pixel values"""
    return model.predict_proba(pred)

# print(make_prediction([data.iloc[2001].drop("Label")]))

# Save the model in a .pkl
import pickle

with open('model.pkl','wb') as f:
    pickle.dump(model, f)

# Code to generate gradio webpage to actively test model. Not enough customization to satisfy me.
# def classify(input):
#     data = input.reshape(1, -1)
#     prediction = model.predict_proba(data).tolist()[0]
#     return {f"{objects[i]} ({i})": prediction[i] for i in range(len(objects))}

# label = gr.outputs.Label(num_top_classes=len(objects), type="confidences")
# interface = gr.Interface(fn=classify, inputs="sketchpad", outputs=label, live=True)
# interface.launch()



"""
Dash Components: Sliders, checkboxes, dropdowns, anything that would normally be a component on a web page
Plotly Graphs: Any graphs created by plotly to visualize data
The Callback: Connects graphs and components to create interaction

Cool idea: Display objects as labels for a bar chart, where there is no movement within the bar chart (no swapping places) just the bars moving up and down based on the prediction values.



"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash_canvas import DashCanvas

app = dash.Dash(__name__)

# -------------------------------------------------------
# App layout

app.layout = html.Div([
    
    html.H1("Ths is a header wooohooooo!!", style={'text-align': 'center'}),

    dcc.Dropdown(id="select_num",
    options = [
        {"label": "5", "value": 5},
        {"label": "7", "value": 7}],
    multi = False,
    value = 5,
    style = {'width': '40%'}
    ),

    html.H5("Press down the left mouse button and draw inside the cnavas!"),

    # DashCanvas(id="doodle_canvas",
    # tool="pencil",
    # lineWidth=15,
    # lineColor="black",
    # width=500,
    # height=1000,
    # hide_buttons=["zoom", "pan", "line", "pencil", "rectangle", "undo", "select", "save"]),


    html.Div(id="output_container", children=[]),
    html.Br(), 


])

if __name__ == "__main__":
    app.run_server(debug=True)