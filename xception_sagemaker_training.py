import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_SIZE = (299, 299)


class EarlyStoppingAtMaxValAccuracy(tf.keras.callbacks.Callback):
    """
    Stop training when the val accuracy is no longer improving.

    Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
    """
    def __init__(self, patience=0):
        super(EarlyStoppingAtMaxValAccuracy, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.wait = None
        self.stopped_epoch = None
        self.best_train = None
        self.best_val = None

    def on_train_begin(self, logs=None):
        # The number of epochs waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_train = 0
        self.best_val = 0

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        print(val_accuracy)
        train_accuracy = logs.get("accuracy")
        print(train_accuracy)

        if np.greater(val_accuracy, self.best_val) and val_accuracy + 0.005 > train_accuracy:
            print("Updating best weights")
            self.best_train = train_accuracy
            self.best_val = val_accuracy
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
            print("The best validation accuracy and corresponding training accuracy were:")
            print("val_accuracy: {}".format(self.best_val))
            print("accuracy: {}".format(self.best_train))


def load_dataset(path, batch_size):
    print(path)
    dataset = image_dataset_from_directory(
        path,
        shuffle=True,
        batch_size=batch_size,
        image_size=IMG_SIZE
    )

    return dataset


def model_training(train_dataset, validation_dataset, fine_tune_lr, fine_tune_epochs, fine_tune_layer):

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        preprocess_input = tf.keras.applications.xception.preprocess_input
        img_shape = IMG_SIZE + (3,)
        base_model = tf.keras.applications.Xception(
            include_top=False, weights="imagenet", input_shape=img_shape
        )

        base_model.trainable = False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1)

        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    # Initial training of the final layer of the model while keeping the rest of the model layers frozen
    initial_epochs = 5
    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)

    with mirrored_strategy.scope():
        base_model.trainable = True

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_layer]:
            layer.trainable = False

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(lr=fine_tune_lr),
            metrics=['accuracy']
        )

        total_epochs = initial_epochs + fine_tune_epochs

    # Train all unfrozen layers of the model together
    model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_dataset,
        callbacks=[EarlyStoppingAtMaxValAccuracy(patience=2)]
    )
    return model


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default, this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])

    # Hyper-parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=0.00001)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layer", type=int, default=100)
    parser.add_argument("--drop-out-percentage", type=float, default=0.1)
    parser.add_argument("--random-rotation", type=float, default=0.2)
    return parser.parse_known_args()


if __name__ == "__main__":
    print("Passing args")
    args, unknown = _parse_args()

    print("loading training data")
    training_data = load_dataset(args.train, args.batch_size)
    val_data = load_dataset(args.validation, args.batch_size)

    print("training model")
    trained_model = model_training(
        training_data, val_data, args.fine_tune_learning_rate, args.fine_tune_epochs, args.fine_tune_layer
    )
    trained_model.save(args.sm_model_dir + "/1")
