
import tensorflow as tf

Mnist=tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest)=Mnist.load_data()

xtrain=xtrain/255
xtest=xtest/255

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)])

predictions = model(xtrain[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(ytrain[:1], predictions).numpy()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=5)

model.evaluate(xtest,  ytest, verbose=2)

probability_model = tf.keras.Sequential([ model, tf.keras.layers.Softmax()])

probability_model(xtest[:5])