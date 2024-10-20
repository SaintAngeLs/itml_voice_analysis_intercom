from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Input, Add, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def residual_block(x, filters, kernel_size=(3, 3)):
    """Residual block with two convolutional layers."""
    shortcut = x  # Save the input tensor

    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def create_cnn_model(input_shape):
    """Create a CNN model with residual blocks."""
    inputs = Input(shape=input_shape)

    # Initial convolutional block
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 1
    x = residual_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 2
    x = residual_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 3
    x = residual_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Add more residual blocks if the dataset is large and training time allows it
    # x = residual_block(x, 512)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # Global Average Pooling (reduces parameter count while keeping spatial info)
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)

    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
