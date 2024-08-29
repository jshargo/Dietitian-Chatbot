from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import date
import requests

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///dietbot.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'
app.app_context().push()

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define the database models
class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    activity_level = db.Column(db.String(50), nullable=False)
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

    user = UserProfile(name=name, password=password, age=age, sex=sex, height=height, weight=weight, activity_level=activity_level)
    db.session.add(user)
    db.session.commit()

    session['user_id'] = user.id
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    user = UserProfile.query.get(user_id)
    today = date.today()

    daily_nutrients = DailyNutrientIntake.query.filter_by(user_id=user_id, date=today).all()
    return render_template('dashboard.html', user=user, daily_nutrients=daily_nutrients)

@app.route('/add_dish')
def add_dish():
    return render_template('add_dish.html')

@app.route('/submit_dish', methods=['POST'])
def submit_dish():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    dish_name = request.form['dish_name']

    # Call the API to get the nutritional info
    response = requests.get(f'https://api.api-ninjas.com/v1/nutrition?query={dish_name}', headers={'X-Api-Key': 'iVUT3BZ+/515ojicGBCKcQ==gMVpcwVlveV3Jvp1'})
    nutrition_data = response.json()

    if nutrition_data:
        # Extract the nutrient data, default to 0 if the value is non-numeric
        data = nutrition_data[0]

        def safe_float(value):
            try: 
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        calories = safe_float(data.get('calories', 0))
        protein = safe_float(data.get('protein_g', 0))
        fat = safe_float(data.get('fat_total_g', 0))
        carbs = safe_float(data.get('carbohydrates_total_g', 0))
        fiber = safe_float(data.get('fiber_g', 0))
        sodium = safe_float(data.get('sodium_mg', 0))

        # Create a new DailyNutrientIntake record
        new_intake = DailyNutrientIntake(user_id=user_id, date=date.today(), dish_name=dish_name,
                                         calories=calories, protein=protein, fat=fat,
                                         carbs=carbs, fiber=fiber, sodium=sodium)
        db.session.add(new_intake)
        db.session.commit()

        flash('Dish added successfully!', 'success')
    else:
        flash('Failed to retrieve nutritional information.', 'danger')

    return redirect(url_for('dashboard'))


@app.route('/show_comparison')
def show_comparison():
    user_id = session.get('user_id')
    user = UserProfile.query.get(user_id)
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

    return render_template('show_comparison.html', daily_nutrients=daily_nutrients, comparison=comparison)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create new tables
    app.run(debug=True, port=8000)
