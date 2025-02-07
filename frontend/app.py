import sys
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import date
import requests
import logging
from dotenv import load_dotenv
import os
from sqlalchemy.types import TypeDecorator, TEXT
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
BACKEND_URL = "http://localhost:8000"

# Example usage
logger.info("Flask app is starting...")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dietbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'
app.app_context().push()

db = SQLAlchemy(app)

# Add this custom type for JSON/List storage
class ListType(TypeDecorator):
    impl = TEXT

    def process_bind_param(self, value, dialect):
        if value is None:
            return '[]'
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return []
        return json.loads(value)

# Define the database models
class UserProfile(db.Model):
    __tablename__ = "user_profile"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    activity_level = db.Column(db.String(50), nullable=False)
    # Modified columns to use ListType
    allergies = db.Column(ListType, default=[])
    likes = db.Column(ListType, nullable=False, default=[])
    dislikes = db.Column(ListType, nullable=False, default=[])
    diet = db.Column(db.String(50), nullable=False)
    goal = db.Column(db.String(50), nullable=False)
    daily_nutrient_intake = db.relationship('DailyNutrientIntake', backref='user', lazy=True)
    

class DailyNutrientIntake(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_profile.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    dish_name = db.Column(db.String(100), nullable=False)
    calories = db.Column(db.Float, default=0)
    protein = db.Column(db.Float, default=0)
    fat = db.Column(db.Float, default=0)
    carbs = db.Column(db.Float, default=0)
    fiber = db.Column(db.Float, default=0)
    sodium = db.Column(db.Float, default=0)

# Define constant RDI values
RDI_VALUES = {
    'Male': {
        'Sedentary': {
            'calories': 2500,
            'protein': 56,
            'fat': 70,
            'carbs': 310,
            'fiber': 30,
            'sodium': 2300},
        'Moderately active': {
            'calories': 2700,
            'protein': 56,
            'fat': 80,
            'carbs': 350,
            'fiber': 30,
            'sodium': 2300},
        'Active': {
            'calories': 3000,
            'protein': 56,
            'fat': 90,
            'carbs': 400,
            'fiber': 30,
            'sodium': 2300}
    },
    'Female': {
        'Sedentary': {
            'calories': 2000,
            'protein': 46,
            'fat': 60,
            'carbs': 260,
            'fiber': 25,
            'sodium': 2300},
        'Moderately active': {
            'calories': 2200,
            'protein': 46,
            'fat': 70,
            'carbs': 300,
            'fiber': 25,
            'sodium': 2300},
        'Active': {
            'calories': 2400,
            'protein': 46,
            'fat': 80,
            'carbs': 340,
            'fiber': 25,
            'sodium': 2300}
    }
}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    password = request.form['password']
    user = UserProfile.query.filter_by(name=name, password=password).first()

    if user:
        session['user_id'] = user.id
        return redirect(url_for('dashboard'))
    else:
        return "Invalid credentials"

@app.route('/create_profile', methods=['POST'])
def create_profile():
    name = request.form['name']
    password = request.form['password']
    age = request.form['age']
    sex = request.form['sex']
    height = request.form['height']
    weight = request.form['weight']
    activity_level = request.form['activity_level']
    # Convert to lists and clean the data
    allergies = [x.strip() for x in request.form['allergies'].split(',') if x.strip()]
    likes = [x.strip() for x in request.form['likes'].split(',') if x.strip()]
    dislikes = [x.strip() for x in request.form['dislikes'].split(',') if x.strip()]
    diet = request.form['diet']
    goal = request.form['goal']

    user = UserProfile(
        name=name,
        password=password,
        age=age,
        sex=sex,
        height=height,
        weight=weight,
        activity_level=activity_level,
        allergies=allergies,  # Now storing as list
        likes=likes,          # Now storing as list
        dislikes=dislikes,    # Now storing as list
        diet=diet,
        goal=goal
    )
    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    user = db.session.get(UserProfile, user_id)
    today = date.today()

    daily_nutrients = DailyNutrientIntake.query.filter_by(user_id=user_id, date=today).all()
    return render_template('chat.html', user=user, daily_nutrients=daily_nutrients)

@app.route('/add_dish')
def add_dish():
    return render_template('add_dish.html')

'''
To be implemented
'''
@app.route('/submit_dish', methods=['POST'])
def submit_dish():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    dish_name = request.form['dish_name']

    # Retrieve Nutritionix credentials from environment variables
    nutritionix_app_id = os.getenv("NUTRITIONIX_APP_ID")
    nutritionix_api_key = os.getenv("NUTRITIONIX_API_KEY")

    # Prepare the payload and headers for the Nutritionix Natural Language API call
    payload = {"query": dish_name}
    headers = {
        "x-app-id": nutritionix_app_id,
        "x-app-key": nutritionix_api_key,
        "Content-Type": "application/json"
    }

    try:
        # Make a POST request to the Nutritionix endpoint
        response = requests.post(
            "https://trackapi.nutritionix.com/v2/natural/nutrients",
            headers=headers,
            json=payload
        )
    except requests.RequestException as e:
        flash("Error connecting to Nutritionix API.", "danger")
        app.logger.error(f"Nutritionix API request error: {e}")
        return redirect(url_for('dashboard'))

    if response.status_code != 200:
        flash("Error retrieving nutritional information.", "danger")
        app.logger.error(f"Nutritionix API error: {response.status_code} - {response.text}")
        return redirect(url_for('dashboard'))

    nutrition_data = response.json()

    # Ensure that the response contains nutritional data
    if "foods" not in nutrition_data or len(nutrition_data["foods"]) == 0:
        flash("No nutritional information found.", "danger")
        return redirect(url_for('dashboard'))

    # Extract data from the first food item returned
    food = nutrition_data["foods"][0]

    def safe_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    calories = safe_float(food.get("nf_calories", 0))
    protein  = safe_float(food.get("nf_protein", 0))
    fat      = safe_float(food.get("nf_total_fat", 0))
    carbs    = safe_float(food.get("nf_total_carbohydrate", 0))
    fiber    = safe_float(food.get("nf_dietary_fiber", 0))
    sodium   = safe_float(food.get("nf_sodium", 0))

    
    

    # Create a new DailyNutrientIntake record using the extracted values
    new_intake = DailyNutrientIntake(
        user_id=user_id,
        date=date.today(),
        dish_name=dish_name,
        calories=calories,
        protein=protein,
        fat=fat,
        carbs=carbs,
        fiber=fiber,
        sodium=sodium
    )
    db.session.add(new_intake)
    db.session.commit()

    flash('Dish added successfully!', 'success')
    return redirect(url_for('dashboard'))

'''
To be implemented
'''
@app.route('/show_comparison')
def show_comparison():
    user_id = session.get('user_id')
    user = db.session.get(UserProfile, user_id)
    today = date.today()

    daily_nutrients = DailyNutrientIntake.query.filter_by(user_id=user_id, date=today).all()
    rdi = RDI_VALUES.get(user.sex, {}).get(user.activity_level, {})

    comparison = {
        'Calories': (sum(dish.calories for dish in daily_nutrients), rdi.get('calories', 0)),
        'Protein (g)': (sum(dish.protein for dish in daily_nutrients), rdi.get('protein', 0)),
        'Fat (g)': (sum(dish.fat for dish in daily_nutrients), rdi.get('fat', 0)),
        'Carbohydrates (g)': (sum(dish.carbs for dish in daily_nutrients), rdi.get('carbs', 0)),
        'Fiber (g)': (sum(dish.fiber for dish in daily_nutrients), rdi.get('fiber', 0)),
        'Sodium (mg)': (sum(dish.sodium for dish in daily_nutrients), rdi.get('sodium', 0))
    }

    return render_template('show_comparison.html', 
                         user=user,
                         daily_nutrients=daily_nutrients, 
                         comparison=comparison)

@app.route("/ask", methods=["POST"])
def ask():
    """Handle chat queries and get intent classification from backend"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401
        
    query = request.form.get("query", "")
    
    logger.info(f"Received query from user {user_id}: {query}")
    
    try:
        # Get user profile data to provide context
        user = db.session.get(UserProfile, user_id)
        today = date.today()
        daily_nutrients = DailyNutrientIntake.query.filter_by(user_id=user_id, date=today).all()
        
        # Prepare payload with user context
        payload = {
            "user_id": user_id,
            "query": query,
            "context": {
                "user_profile": {
                    "age": user.age,
                    "sex": user.sex,
                    "height": user.height,
                    "weight": user.weight,
                    "activity_level": user.activity_level,
                    "allergies": user.allergies,
                    "likes": user.likes,
                    "dislikes": user.dislikes,
                    "diet": user.diet,
                    "goal": user.goal
                },
                "daily_nutrients": [
                    {
                        "dish_name": n.dish_name,
                        "calories": n.calories,
                        "protein": n.protein,
                        "fat": n.fat, 
                        "carbs": n.carbs,
                        "fiber": n.fiber,
                        "sodium": n.sodium
                    } for n in daily_nutrients
                ]
            }
        }
        
        logger.info(f"Sending request to backend at {BACKEND_URL}")
        resp = requests.post(
            f"{BACKEND_URL}/query",
            json=payload,
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Received response from backend: {data}")
            
            # Fix: Properly extract context_used from response
            return jsonify({
                "reasoning": data.get("reasoning", ""),
                "answer": data.get("final_answer", ""),
                "context_used": data.get("context_used", ""),
                "response": data.get("final_answer", "")
            })
        else:
            error_msg = f"Backend error: {resp.status_code}"
            logger.error(error_msg)
            return jsonify({"error": error_msg, "response": "Sorry, I encountered an error processing your request."}), 200
            
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error to backend: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "response": "Sorry, I'm having trouble connecting to the server."}), 200
    except requests.Timeout:
        error_msg = "Backend request timed out"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "response": "Sorry, the request timed out. Please try again later."}), 200
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "response": "Sorry, something went wrong."}), 200



@app.route('/logout')
def logout():
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = db.session.get(UserProfile, session['user_id'])
    
    if request.method == 'POST':
        user.height = request.form['height']
        user.weight = request.form['weight']
        user.activity_level = request.form['activity_level']
        # Convert to lists
        user.allergies = [x.strip() for x in request.form['allergies'].split(',') if x.strip()]
        user.likes = [x.strip() for x in request.form['likes'].split(',') if x.strip()]
        user.dislikes = [x.strip() for x in request.form['dislikes'].split(',') if x.strip()]
        diet_option = request.form['diet']
        goal_option = request.form['goal']
        valid_diets = ["No Restriction", "Vegetarian", "Vegan", "Gluten-Free", "Lactose-Intolerant", "Halal", "Pescetarian"]
        valid_goals = ["Maintain Weight", "Lose Weight", "Gain Weight"]
        if diet_option not in valid_diets or goal_option not in valid_goals:
            flash('Invalid diet or goal.', 'danger')
            return redirect(url_for('profile'))
        user.diet = diet_option
        user.goal = goal_option
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('profile.html', user=user)

@app.route("/user_context/{user_id}", methods=["GET"])
def get_user_context_new(user_id: int):
    user = db.session.get(UserProfile, user_id)
    if not user:
        return jsonify({"error": "No user found"}), 404
    '''
    To be implemented: recommended calories for dishes
    '''
    return jsonify({
        "name": user.name,
        "age": user.age,
        "sex": user.sex,
        "height": user.height,
        "weight": user.weight,
        "activity_level": user.activity_level,
        "allergies": user.allergies,
        "likes": user.likes,
        "dislikes": user.dislikes,
        "diet": user.diet,
        "goal": user.goal
    }), 200

if __name__ == '__main__':
    with app.app_context():
        #drop all tables
        db.create_all()
        app.run(debug=True, host='localhost', port=8001, use_reloader=True)
