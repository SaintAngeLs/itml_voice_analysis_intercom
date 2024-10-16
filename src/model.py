from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, Input, Add
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=(3, 3)):
    """A basic residual block with convolutional layers."""
    shortcut = x  # Save the input tensor

    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # If the input and output filters are different, apply a 1x1 convolution to the shortcut to match dimensions
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    # Adding the shortcut
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

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Binary classification output with sigmoid activation
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
