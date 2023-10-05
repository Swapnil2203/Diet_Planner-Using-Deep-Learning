import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

df = pd.read_csv("output.csv")

model= keras.models.load_model('model.h5')

def generate_diet_plan(predictions, required_calories):
    foods = []
    calorie_intake = 0
    while calorie_intake < required_calories:
        top_3 = np.argsort(predictions[0])[::-1][:3]
        for i in top_3:
            food_calories = df.iloc[i]['Calories']
            if calorie_intake + food_calories <= required_calories:
                foods.append(df.iloc[i]['Description'])
                calorie_intake += food_calories
        predictions[0][top_3] = -1 # set the top 3 predictions to -1 to avoid selecting them again
    return foods

user_input = [2850, 356, 143, 95] # user inputs their target daily calories, carbs, protein, and fat
required_calories = user_input[0] # set the required calories to the user's target daily calories
user_input = np.array(user_input).reshape(1, -1) # convert user input to a 2D numpy array with a single row
predictions = model.predict(user_input)
recommended_foods = generate_diet_plan(predictions, required_calories)
print('Recommended Foods:', recommended_foods)