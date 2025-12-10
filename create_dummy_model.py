import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import numpy as np

def create_and_save_model():
    print("creating dummy model...")
    # Use MobileNetV2 as a base for a lightweight model
    # weights=None to avoid downloading from internet (which caused SSL error)
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2), # Prevent overfitting
        Dense(5, activation='softmax') # 5 classes (S, A, B, C, D)
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Model structure:")
    model.summary()

    # Save the model
    model_path = 'horse_body_model.h5'
    model.save(model_path)
    print(f"Dummy model saved to {model_path}")

if __name__ == "__main__":
    create_and_save_model()
