import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import plotly  # Ensure plotly is imported
import json

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Student_Marks.csv")

# Prepare the data for the model
X = data[['time_study', 'number_courses']]
y = data['Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time_study = float(request.form['time_study'])
    number_courses = int(request.form['number_courses'])
    
    # Make a prediction
    predicted_marks = model.predict(np.array([[time_study, number_courses]]))[0]

    # Cap the predicted marks at 100
    capped_marks = min(predicted_marks, 100)

    # Prepare data for plotting
    # Original data plot
    scatter_data = go.Scatter(
        x=data['time_study'],
        y=data['Marks'],
        mode='markers',
        name='Original Data',
        marker=dict(color='blue')
    )

    # Predicted data plot
    predicted_data = go.Scatter(
        x=[time_study],
        y=[capped_marks],  # Use capped_marks instead of predicted_marks
        mode='markers+text',
        name='Predicted Data',
        text=[f'Predicted: {capped_marks:.2f}'],
        textposition='top center',
        marker=dict(color='red', size=10)
    )

    # Create a layout for the graph
    layout = go.Layout(
        title='Marks vs Time Studied',
        xaxis=dict(title='Time Studied (hours)'),
        yaxis=dict(title='Marks'),
        showlegend=True
    )

    # Create figure
    fig = go.Figure(data=[scatter_data, predicted_data], layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify(predicted_marks=capped_marks, plot=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
