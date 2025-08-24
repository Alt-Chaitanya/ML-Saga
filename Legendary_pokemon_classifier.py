'''starting off with custom made pokemon dataset. 
mlday1.....will explain this code and how scikit learn functions affect total output. 
also will understand all algos step by step in coming days in the form of individual programs.'''




import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# I made custom Data
data = {
    'Name': [
        'Pikachu', 'Charizard', 'Bulbasaur', 'Squirtle', 'Mewtwo', 'Gengar', 'Eevee', 'Snorlax',
        'Dragonite', 'Lucario', 'Articuno', 'Zapdos', 'Moltres', 'Gyarados', 'Lapras', 'Jigglypuff',
        'Onix', 'Machamp', 'Alakazam', 'Ditto', 'Blastoise', 'Venusaur', 'Tyranitar', 'Scizor',
        'Celebi', 'Entei', 'Raikou', 'Suicune', 'Registeel', 'Latios', 'Latias', 'Rayquaza',
        'Kyogre', 'Groudon', 'Deoxys', 'Lugia', 'Ho-oh', 'Manaphy', 'Greninja','meowth'
    ],
    'Attack': [
        55, 84, 49, 48, 110, 65, 55, 110,
        134, 110, 85, 90, 100, 125, 85, 45,
        45, 130, 50, 48, 83, 82, 134, 130,
        100, 115, 85, 75, 75, 90, 80, 150,
        100, 150, 150, 90, 130, 100, 103,90
    ],
    'Defense': [
        40, 78, 49, 65, 90, 60, 50, 65,
        95, 70, 100, 85, 90, 79, 80, 20,
        160, 80, 45, 48, 100, 83, 110, 100,
        100, 85, 75, 115, 150, 80, 90, 90,
        90, 140, 50, 130, 90, 100, 67,90
    ],
    'Speed': [
        90, 100, 45, 43, 130, 110, 55, 30,
        80, 90, 85, 100, 90, 81, 60, 20,
        70, 55, 120, 48, 78, 80, 61, 65,
        100, 100, 115, 85, 50, 110, 110, 95,
        90, 90, 150, 110, 90, 122,80,87
    ],
    'Legendary': [
        False, False, False, False, True, False, False, False,
        False, False, True, True, True, False, False, False,
        False, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, False,True
    ]
}
df = pd.DataFrame(data)


X = df[['Attack', 'Defense', 'Speed']] #features(input data for model to learn from.)
Y = df['Legendary'] #labels(target values I want to predict


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# here I chose logistic regression cause it's best for binary classification. 
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train) #train the model. 

# Evaluation metrics...
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# taking User Input  
pokemon = input("Enter Pokémon name: ")
stats = input(f"Enter {pokemon}'s Attack, Defense, Speed (comma-separated): ")
statlist = [int(x) for x in stats.split(",")]

X_new = pd.DataFrame([statlist], columns=['Attack','Defense','Speed'])
Y_pred_new = model.predict(X_new)[0]
prob = model.predict_proba(X_new)[0][1]

print(f"Attack, Defense, Speed: {statlist}")
print(f"Is {pokemon} Legendary? {Y_pred_new} (Probability: {prob:.2f})")