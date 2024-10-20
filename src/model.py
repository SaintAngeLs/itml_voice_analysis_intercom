from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, Input, Add, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=(3, 3)):
    shortcut = x  # Save the input tensor

    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Shortcut adjustment
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    return x

def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)

    # Initial convolutional block
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = residual_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # GlobalAveragePooling instead of Flatten
    x = GlobalAveragePooling2D()(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Binary classification output
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
