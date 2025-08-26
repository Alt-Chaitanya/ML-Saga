# mld2: trying to write Logistic Regression from scratch to understand it better
import numpy as np

# ----------------------------
# 1. making small Dataset which was earlier used in mld1
# ----------------------------
# Features = [Attack, Defense]
X = np.array([
    [120,130],
    [100,125], 
    [60,70], 
    [80,60], 
    [95,90]
])
Y = np.array([1,1,0,0,0])    # 1 = Legendary, 0 = Non-legendary

# ----------------------------
# 2. Standardization (for stability)
# ----------------------------
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-12  # avoid division by 0
X_stdized = (X - X_mean) / X_std

def standardize_new(x_row):
    return (x_row - X_mean) / X_std

# ----------------------------
# 3. Sigmoid function
# ----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# ----------------------------
# 4. Prediction for initial weights
# ----------------------------
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# ----------------------------
# 5. Loss function (Log loss)
# ----------------------------
def compute_log_loss(Y, Y_pred):
    m = Y.shape[0]
    epsilon = 1e-15
    Y_pred = np.clip(Y_pred, epsilon, 1-epsilon)
    log_loss = (-1/m) * np.sum(Y*np.log(Y_pred) + (1-Y)*np.log(1-Y_pred))
    return log_loss

# ----------------------------
# 6.Computing Gradients
# ----------------------------
def compute_gradients(X, Y, Y_pred):
    m = Y.shape[0]
    error = Y_pred - Y
    dw = (1/m) * np.dot(X.T, error)   # gradients for weights
    db = (1/m) * np.sum(error)        # gradient for bias
    return dw, db

# ----------------------------
# 7. Training Loop
# ----------------------------
def train_logistic_regression(X, Y, lr=0.1, epochs=1000):
    weights = np.zeros(X.shape[1])
    bias = 0
    losses = []
   
    for i in range(epochs):
        Y_pred = predict(X, weights, bias)
        loss = compute_log_loss(Y, Y_pred)
        losses.append(loss)

        dw, db = compute_gradients(X, Y, Y_pred)
        weights -= lr * dw
        bias -= lr * db
        
        if i % 50 == 0:
            print(f"epoch:{i}: loss:{loss:.4f}")
            
    return weights, bias, losses

# ----------------------------
# 8. Training with standardized features
# ----------------------------
weights, bias, losses = train_logistic_regression(X_stdized, Y)
print(f"Weights: {weights}")
print(f"Bias: {bias}")

# ----------------------------
# 9. Testing new Pokémon stat
# ----------------------------
attack  = int(input("Please Enter Stat for Attack: "))
defense = int(input("Please Enter Stat for Defense: "))

X_new = np.array([attack, defense], dtype=float)
X_new_std = standardize_new(X_new)

z = np.dot(X_new_std, weights) + bias
p = sigmoid(z)

print("Input (raw):", X_new)
print(f"Logit z = {z:.4f}, Probability Legendary = {p:.6f}")
