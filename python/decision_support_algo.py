import tensorflow as tf

dataset = tf.TensorArray(
    dtype=tf.float32,
    size=0,
    dynamic_size=True,
    clear_after_read=False)
dataset.write(0, [0.1, 0.1])
dataset.write(1, [0.3, 0.3])
dataset.write(2, [0.5, 0.6])
dataset.write(3, [0.4, 0.8]),
dataset.write(4, [0.9, 0.1])
dataset.write(5, [0.75, 0.4])
dataset.write(6, [0.75, 0.9])
dataset.write(7, [0.6, 0.9])
dataset.write(8, [0.6, 0.75])

# rewrite the TensorArray as a Tensor
dataset = dataset.stack()
print(dataset)

preferences = tf.constant([0, 0, 1, 1, 0, 0, 1, 1, 1])
# transform preferences to 1-hot encoding
preferences = tf.one_hot(preferences, 2)

#We have a 2-layer network with an input layer containing 2 neurons, 
# a hidden layer with 3 neurons and an output layer containing 2 neurons.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(2, activation='softmax')])

#Note that we use ReLu activation function in the hidden layer and softmax for the output layer. 
# We have 2 neurons in the output layer since we want to obtain how certain our Neural Network is in 
# its buy/no-buy decision.

# set optimizer learning rate to 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# train the model defining callbacks function
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True
model.fit(dataset, preferences, epochs=10000, callbacks=[myCallback()], shuffle=True)

predProb = model.predict(tf.constant([[0.1, 0.6]]))

print("Probability of buying: ", predProb[0][1])
