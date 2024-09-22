from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from flask import Flask, render_template, request, redirect
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
import json
import base64

from flask import session
from main import get_output
# Import the generate_prompts function from prompts.py
from prompt import generate_prompts
app = Flask(__name__)
@app.route('/save-image', methods=['POST'])
def save_image():
    data = request.get_json()
    image_data = data.get('image')

    if image_data:
        # Extract the base64 part of the image data
        image_data = image_data.split(',')[1]
        image_data = base64.b64decode(image_data)

        # Define the path where you want to save the image
        image_path = os.path.join('static', 'images', 'map_image.png')  # Adjust the path as needed

        # Save the image to the specified path
        with open(image_path, 'wb') as f:
            f.write(image_data)

        return {'message': 'Image saved successfully'}, 200
    return {'message': 'No image data provided'}, 400
prompts = {
    'Basic Information Metrics': {
        'id':'Basic Information Metrics',
        'description': 'Here is basic information about the location you selected, including population, literacy rate, and GDP.',
        'icon': '<i class="fas fa-info-circle"></i>',
        'no':'3'
    },
    'Consumption Metrics': {
        'id':'Consumption Metrics',
        'description': 'These prompts provide insights into the consumption of resources like power, water, and fuel in the area.',
        'icon': '<i class="fas fa-water"></i>',
        'no':'5'
    },
    'Crime Metrics': {
        'id':'Crime Metrics',
        'description': 'Get an overview of crime statistics and major crime categories in the selected city.',
        'icon': '<i class="fas fa-shield-alt"></i>',
        'no':'5'
    },
    'Living Conditions Metrics': {
        'id':'Living Conditions Metrics',
        'description': 'Find out about housing costs, waste management, healthcare services, and education in the area.',
        'icon': '<i class="fas fa-home"></i>',
        'no':'5'
    },
    'Farming Data Metrics': {
        'id':'Farming Data Metrics',
        'description': 'Explore information about agriculture, including arable land, water usage, and employment in farming.',
        'icon': '<i class="fas fa-seedling"></i>',
        'no':'5'
    },
    'Economic Data Metrics': {
        'id':'Economic Data Metrics',
        'description': 'This section covers economic indicators such as GDP, unemployment rate, and income levels.',
        'icon': '<i class="fas fa-dollar-sign"></i>',
        'no':'5'
    },
    'Historic Disasters Metrics': {
        'id':'Historic Disasters Metrics',
        'description': 'Learn about natural disasters that have impacted the area and their consequences.',
        'icon': '<i class="fas fa-exclamation-triangle"></i>',
        'no':'10'
    },
    'Demographic Data Metrics': {
        'id':'Demographic Data Metrics',
        'description': 'Access demographic statistics, including age and gender distribution, migration rates, and literacy.',
        'icon': '<i class="fas fa-users"></i>',
        'no':'5'
    },
    'Industrial Stats Metrics': {
        'id':'Industrial Stats Metrics',
        'description': 'Review data on industrial employment, output value, growth rates, and environmental impact.',
        'icon': '<i class="fas fa-industry"></i>',
        'no':'5'
    },
    'Transportation Stats Metrics': {
        'id':'Transportation Stats Metrics',
        'description': 'Understand transportation patterns, public transit usage, and commute times in the city.',
        'icon': '<i class="fas fa-bus"></i>',
        'no':'5'
    },
    'Biodiversity Stats Metrics': {
        'id':'Biodiversity Stats Metrics',
        'description': 'Discover information about plant and animal species, habitats, and biodiversity index.',
        'icon': '<i class="fas fa-leaf"></i>',
        'no':'5'
    },
    'Geographic Stats Metrics': {
        'id':'Geographic Stats Metrics',
        'description': 'Examine geographical features, climate zones, and natural resources in the location.',
        'icon': '<i class="fas fa-globe"></i>',
        'no':'5'
    },
    'Telecom Stats Metrics': {
        'id':'Telecom Stats Metrics',
        'description': 'Access data on internet access, mobile coverage, and digital literacy in the area.',
        'icon': '<i class="fas fa-mobile-alt"></i>',
        'no':'5'

    },
    'Public Services Stats Metrics': {
        'id':'Public Services Stats Metrics',
        'description': 'Find out about the availability of healthcare facilities, schools, and emergency services.',
        'icon': '<i class="fas fa-hospital"></i>',
        'no':'5'
    },
    'Noise Levels Stats Metrics': {
        'id':'Noise Levels Stats Metrics',
        'description': 'Learn about average noise levels, sources of noise pollution, and quiet zones in the city.',
        'icon': '<i class="fas fa-volume-up"></i>',
        'no':'5'
    }
}
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production
class LocationForm(FlaskForm):
    locality = StringField('Locality (Optional)')
    city = StringField('City', validators=[DataRequired()])
    state = StringField('State', validators=[DataRequired()])
    country = StringField('Country', validators=[DataRequired()])
    submit = SubmitField('Generate Prompts')
# Home route
@app.route('/')
def index():
    return render_template('index.html')

def format_location(location):
   
    parts = [
        location.get('locality'),
        location.get('city'),
        location.get('state'),
        location.get('country')
    ]
    # Filter out None or empty strings
    parts = [part for part in parts if part]
    return ', '.join(parts)
@app.route('/input', methods=['GET', 'POST'])
def input_location():
    form = LocationForm()
    api_key = '4aa007acec6344a68e0aa466fcf6dc0b'
    if not api_key:
        flash('API key is not set. Please contact the administrator.', 'danger')
        return redirect(url_for('index'))

    if form.validate_on_submit():
        locality = form.locality.data
        city = form.city.data
        state = form.state.data
        country = form.country.data

        location = {
            'locality': locality,
            'city': city,
            'state': state,
            'country': country
        }

        formatted_location = format_location(location)
        current_year = datetime.today().year

        # Store data in session
        session['formatted_location'] = formatted_location
        session['location'] = location
        session['current_year'] = current_year

        return redirect(url_for('show_prompts'))

    return render_template('input.html', form=form, Api_key=api_key)
# Ensure this template exists

# Features route
@app.route('/features')
def features():
    return render_template('features.html')  # Ensure this template exists

# Projects route
@app.route('/projects')
def projects():
    return render_template('projects.html')  # Ensure this template exists

# Documentation route
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')  # Ensure this template exists

# Blog route (add if needed)
@app.route('/blog')
def blog():
    return render_template('blog.html')  # Create this template if needed

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')  # Ensure this template exists

# Quote request route (form submission)
@app.route('/quote', methods=['POST'])
def quote():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    message = request.form.get('message')

    # TODO: Process the form data (e.g., send an email or save to a database)

    flash('Your quote request has been sent successfully. Thank you!', 'success')
    return redirect(url_for('index'))

@app.route('/prompts', methods=['GET', 'POST'])
def show_prompts():
    if request.method == 'POST':
        # Extract data from the form
        locality = request.form.get('locality', '').strip()
        city = request.form.get('city', '').strip()
        state = request.form.get('state', '').strip()
        country = request.form.get('country', '').strip()

        # Validate or process the data as needed
        # For example, ensure that at least one field is provided
        if not any([locality, city, state, country]):
            # Handle the error: no data provided
            # You can redirect back with a message or render the template with an error
            return render_template('prompts.html', prompts=prompts, location={}, 
                                   formatted_location="Unknown Location", 
                                   current_year=datetime.today().year,
                                   error="No location data provided.")

        # Construct the location dictionary
        location = {
            'locality': locality,
            'city': city,
            'state': state,
            'country': country
        }

        # Optionally, you can store this in the session if needed
        # session['location'] = location

        # Now render the prompts.html with the provided location
        return render_template(
            'prompts.html',
            prompts=prompts,
            location=location,
            formatted_location=format_location(location),
            current_year=datetime.today().year
        )
    else:
        # For GET requests, you might want to redirect or show a default page
        # Here, we'll just render prompts.html with default data
        location = {
            'locality': 'Unknown',
            'city': 'Unknown',
            'state': 'Unknown',
            'country': 'Unknown'
        }
        return render_template(
            'prompts.html',
            prompts=prompts,
            location=location,
            formatted_location=format_location(location),
            current_year=datetime.today().year
        )



   
@app.route('/get-output', methods=['GET'])
def get_output_route():
    prompt_id = request.args.get('prompt_id')
    location = json.loads(request.args.get('location'))
    top_k_s = request.args.get('top_k_s')
    top_k_s = int(top_k_s)
    

    # Call your output function
    output_data = get_output(prompt_id, location, top_k_s)

    return jsonify({'output': output_data})

if __name__ == '__main__':
    app.run(debug=True)
