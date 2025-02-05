BEGIN TRANSACTION;

CREATE TABLE user_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    age INTEGER NOT NULL,
    sex VARCHAR(10) NOT NULL,
    height INTEGER NOT NULL,
    weight INTEGER NOT NULL,
    allergies VARCHAR(100),
    activity_level VARCHAR(50) NOT NULL
);

CREATE TABLE daily_nutrient_intake (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date DATE NOT NULL,
    dish_name VARCHAR(100) NOT NULL,
    calories FLOAT DEFAULT 0,
    protein FLOAT DEFAULT 0,
    fat FLOAT DEFAULT 0,
    carbs FLOAT DEFAULT 0,
    fiber FLOAT DEFAULT 0,
    sodium FLOAT DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES user_profile (id)
);

COMMIT; 