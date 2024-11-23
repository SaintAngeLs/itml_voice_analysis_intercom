from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Dense,
    Input,
    Add,
    GlobalAveragePooling2D,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class ResidualBlock:
    """Class to construct a residual block with configurable parameters."""

    def __init__(self, filters, kernel_size=(3, 3), regularizer=l2(0.001)):
        """
        Initialize ResidualBlock.

        Parameters:
        - filters: Number of filters for the convolutional layers.
        - kernel_size: Size of the convolutional kernel (default is (3, 3)).
        - regularizer: Regularizer for the convolutional layers (default is L2 regularization).
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.regularizer = regularizer

    def build(self, x):
        """
        Build the residual block.

        Parameters:
        - x: Input tensor.

        Returns:
        - Output tensor after applying the residual block.
        """
        shortcut = x

        # First convolutional layer
        x = Conv2D(
            self.filters, self.kernel_size, padding="same", kernel_regularizer=self.regularizer
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Second convolutional layer
        x = Conv2D(
            self.filters, self.kernel_size, padding="same", kernel_regularizer=self.regularizer
        )(x)
        x = BatchNormalization()(x)

        # Adjust the shortcut dimensions if necessary
        if shortcut.shape[-1] != self.filters:
            shortcut = Conv2D(
                self.filters, (1, 1), padding="same", kernel_regularizer=self.regularizer
            )(shortcut)

        # Combine shortcut and output
        x = Add()([x, shortcut])
        x = Activation("relu")(x)
        return x


class CNNModel:
    """Class to construct the CNN model with residual blocks."""

    def __init__(self, input_shape, num_classes=1, dropout_rate=0.7, regularizer=l2(0.001)):
        """
        Initialize CNNModel.

        Parameters:
        - input_shape: Shape of the input data (e.g., (height, width, channels)).
        - num_classes: Number of output classes (default is 1 for binary classification).
        - dropout_rate: Dropout rate for the fully connected layers (default is 0.7).
        - regularizer: Regularizer for convolutional layers (default is L2 regularization).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer

    def build(self):
        """
        Build the CNN model.

        Returns:
        - Keras Model object.
        """
        inputs = Input(shape=self.input_shape)

        # Initial convolutional block
        x = Conv2D(
            64, (3, 3), padding="same", kernel_regularizer=self.regularizer
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Residual blocks
        x = ResidualBlock(64, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = ResidualBlock(128, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # x = ResidualBlock(256, regularizer=self.regularizer).build(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)

        # Fully connected layer
        x = Dense(256, activation="relu", kernel_regularizer=self.regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        outputs = Dense(self.num_classes, activation=activation)(x)

        # Build the model
        model = Model(inputs=inputs, outputs=outputs)
        return model


class CNNModelSimpler:
    """Simpler CNN model with fewer layers and no residual blocks."""

    def __init__(self, input_shape, num_classes=1, dropout_rate=0.5, regularizer=l2(0.001)):
        """
        Initialize CNNModelSimpler.

        Parameters:
        - input_shape: Shape of the input data (e.g., (height, width, channels)).
        - num_classes: Number of output classes (default is 1 for binary classification).
        - dropout_rate: Dropout rate for the fully connected layers (default is 0.5).
        - regularizer: Regularizer for convolutional layers (default is L2 regularization).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer

    def build(self):
        """
        Build the CNN model.

        Returns:
        - Keras Model object.
        """
        inputs = Input(shape=self.input_shape)

        # Simple convolutional block
        x = Conv2D(
            32, (3, 3), padding="same", kernel_regularizer=self.regularizer
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(
            64, (3, 3), padding="same", kernel_regularizer=self.regularizer
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)

        # Fully connected layer
        x = Dense(128, activation="relu", kernel_regularizer=self.regularizer)(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        outputs = Dense(self.num_classes, activation=activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model



class CNNModelMoreAdvanced:
    """More advanced CNN model with additional residual blocks and layers."""

    def __init__(self, input_shape, num_classes=1, dropout_rate=0.7, regularizer=l2(0.001)):
        """
        Initialize CNNModelMoreAdvanced.

        Parameters:
        - input_shape: Shape of the input data (e.g., (height, width, channels)).
        - num_classes: Number of output classes (default is 1 for binary classification).
        - dropout_rate: Dropout rate for the fully connected layers (default is 0.7).
        - regularizer: Regularizer for convolutional layers (default is L2 regularization).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer

    def build(self):
        """
        Build the CNN model.

        Returns:
        - Keras Model object.
        """
        inputs = Input(shape=self.input_shape)

        # Initial convolutional block
        x = Conv2D(
            64, (3, 3), padding="same", kernel_regularizer=self.regularizer
        )(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Additional residual blocks
        x = ResidualBlock(64, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = ResidualBlock(128, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = ResidualBlock(256, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = ResidualBlock(512, regularizer=self.regularizer).build(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)

        # Fully connected layer
        x = Dense(512, activation="relu", kernel_regularizer=self.regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Output layer
        activation = "sigmoid" if self.num_classes == 1 else "softmax"
        outputs = Dense(self.num_classes, activation=activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model


# This is a modelur component class, the main class caller it ony for demonstration purposes
# if __name__ == "__main__":
#     input_shape = (128, 128, 3)  
#     model_builder = CNNModel(input_shape=input_shape, num_classes=1)
#     model = model_builder.build()
#     model.summary()
