from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

def build_and_train_model(X_train, y_train, X_val, y_val):
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build a more optimized model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.00003),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for learning rate reduction and early stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model with data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=[lr_scheduler])

    return model, history