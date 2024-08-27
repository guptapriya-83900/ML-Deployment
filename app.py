from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        



'''Here's a concise summary of how flask is executed:

Running app.py: When you run app.py directly, Python sets the __name__ variable to "__main__" because app.py is the main program being executed.

Flask Initialization: When you create the Flask app with app = Flask(__name__), Flask uses __name__ (which is "__main__" in this case) to figure out the root directory of your project (where app.py is located).

Locating Templates and Static Files: With the root directory known, Flask can locate the templates/ and static/ folders relative to where app.py is. This allows Flask to serve HTML files, CSS, JavaScript, and other static assets correctly.

Handling Requests: Depending on whether a request is a GET or POST request, Flask routes the request to the appropriate function in app.py, executing the corresponding code.

So, in short, __main__ helps Flask understand where your application is running from, enabling it to find necessary resources 
and handle incoming requests properly. 

Routing: Routing refers to mapping URLs to functions in your Flask application. 
A route is defined using a decorator (@app.route). When a user visits a URL, Flask looks for a route that matches the URL 
and then calls the associated function to handle the request.

Request and Response: Flask handles incoming web requests and returns responses. 
The request object contains data sent by the client (like form inputs or query parameters),and the response is what Flask sends 
back to the client (like HTML, JSON, or other data).
    Form Data (Key-Value Pairs): When a user submits a form on a webpage, the data is sent as key-value pairs via a POST request.
    You retrieve this data in Flask using request.form, which is a dictionary-like object containing the form data.
    Example:
    gender = request.form.get('gender')
    age = request.form.get('age')

    Here, gender and age are keys, and their corresponding values are what the user entered in the form.

'''
'''
Flask Workflow:

1. Create an app.py File: Start by creating a Python file named app.py, which will be the main file for your Flask application.

2. Initialize the Flask Application: Inside app.py, initialize the Flask app with app = Flask(__name__). This sets up the application and allows Flask to figure out where the necessary templates and static files (like HTML, CSS, and JavaScript) are located relative to the script.

3. Define Routes and Functions:

    1. Home Page Route: Create a function that handles the home page, where you can introduce your website. Use the @app.route('/') decorator to map this function to the root URL (/). The function will return a rendered HTML template, typically using render_template('home.html').

    2. Prediction Page Route: Create another route for the prediction page where users can input data and get results. This function will handle two types of HTTP requests:

    3. GET Request: When a user visits the prediction page, the GET request simply renders the prediction page (e.g., render_template('prediction.html')).

    4. POST Request: When a user submits the form on the prediction page, the POST request will:
                    1. Capture the input data from the form.
                    2. Convert the data into a format (like a Pandas DataFrame) suitable for your machine learning model.
                    3. Pass the formatted data to the model to generate predictions.
                    4. Use render_template to display the prediction results on the web page.

4. Run the Application: Finally, ensure your app runs when you execute app.py by including the following at the end of your script:

        if __name__ == "__main__":
            app.run(debug=True)
        This line ensures that your Flask application starts when you run app.py directly.

'''
'''
Things to keep in mind:
1. The name attribute in the HTML form element is what connects the form field to the Flask backend.
   When you use request.form.get('gender') in your Flask code, the 'gender' inside the parentheses must match the name="gender" 
   in your HTML form.
2. The values provided in the <option> elements of your HTML form should match exactly with the values in your training dataset, 
   including being case-sensitive. This is crucial for ensuring that the input data passed to your machine learning model is 
   consistent with what the model was trained on.
'''

'''
Website Interaction with Flask:
1. Starting the Flask App
    You start your Flask application by running python app.py.
    Flask initializes and starts a web server, which listens for incoming requests (usually at http://127.0.0.1:5000/).

2. Visiting the Homepage (/)
    URL Accessed: A user opens a web browser and goes to http://127.0.0.1:5000/.

    Flask Route:
    @app.route('/')
    def index():
        return render_template('home.html')

    Flow:
    The browser sends a GET request to http://127.0.0.1:5000/.
    Flask checks if there’s a route that matches /.
    The index function is found, and Flask executes it.
    The index function renders the home.html template and sends it back to the browser.
    The user sees the content of home.html in their browser.

3. User Navigates to the Prediction Page or Submits the Form
    URL Accessed: Suppose the user is on a page where they submit a form to http://127.0.0.1:5000/predictdata. This can happen either by navigating directly to http://127.0.0.1:5000/predictdata (GET request) or by submitting a form from home.html (POST request).
    Flask Route:
    @app.route('/predictdata', methods=['GET', 'POST'])
    def predict_datapoint():
    
4. GET Request to /predictdata
    User Action: The user might manually navigate to http://127.0.0.1:5000/predictdata (e.g., by typing it into the browser).

    Flow:
    The browser sends a GET request to http://127.0.0.1:5000/predictdata.
    Flask checks the route @app.route('/predictdata', methods=['GET', 'POST']) and matches it.
    Inside predict_datapoint, Flask checks if request.method == 'GET':.
    Since the method is GET, Flask renders and returns index.html.
    The user sees the content of index.html in their browser.

5. POST Request to /predictdata (Form Submission)
    User Action: The user fills out a form on a page (e.g., home.html) and submits it.
    Form HTML (Example):
    <form action="{{ url_for('predict_datapoint') }}" method="post">
        <!-- form fields -->
        <button type="submit">Submit</button>
    </form>

    Flow:
        The form’s action="{{ url_for('predict_datapoint') }}" points to the /predictdata route, so submitting the form sends a POST request to http://127.0.0.1:5000/predictdata.
        Flask matches the request to the @app.route('/predictdata', methods=['GET', 'POST']) route.
        Inside predict_datapoint, Flask checks if request.method == 'GET':. This is false, so it skips to the else block.
        Inside else Block:
        Data from the form is collected using request.form.get().
        The data is processed (converted into a DataFrame).
        Predictions are made using your model.
        The prediction results are then passed to index.html for rendering.
        
    Render: Flask renders index.html with the prediction results embedded and sends it back to the browser.
        The user sees the results on the updated page.

6. Displaying the Prediction
Final Output:
After the form is submitted and processed, the prediction results are displayed on the page (within index.html).
'''