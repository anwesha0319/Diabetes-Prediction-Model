import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:/Users/anwes/OneDrive/Desktop/templates/finalmod.csv')


# Replace zero values with NaN
cols_with_zero_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Urine']
data[cols_with_zero_values] = data[cols_with_zero_values].replace(0, np.nan)

# Use KNNImputer to fill missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# Convert back to DataFrame
data = pd.DataFrame(data_imputed, columns=data.columns)

# Feature Engineering: Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(data.drop(columns='Outcome'))
X_poly = pd.DataFrame(X_poly)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_poly)

# Separate features and label
y = data['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the ANN model with more hidden layers
def build_ann_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu', kernel_regularizer='l2'))
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Initialize the ANN model
ann_model = build_ann_model(X_train.shape[1])

# Compile the model
ann_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Early stopping, model checkpoint, and learning rate scheduler callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# model_checkpoint = ModelCheckpoint('best_ann_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

# Train the ANN model
history = ann_model.fit(X_train, y_train, epochs=300, batch_size=32, 
                        validation_data=(X_test, y_test), 
                        callbacks=[early_stopping, reduce_lr], verbose=1)

# Evaluate the ANN model
ann_loss, ann_accuracy = ann_model.evaluate(X_test, y_test)
print(f'ANN Model Accuracy: {ann_accuracy:.2f}')

# Plotting accuracy and loss for ANN model
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
model.save('C:/Users/anwes/OneDrive/Desktop/templates/best_ann_model.keras')
