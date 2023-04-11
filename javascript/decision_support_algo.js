
const tf = require('@tensorflow/tfjs');

const X = tf.tensor2d([

  // pink, small

  [0.1, 0.1],
  [0.3, 0.3],
  [0.5, 0.6],
  [0.4, 0.8],
  [0.9, 0.1],
  [0.75, 0.4],
  [0.75, 0.9],
  [0.6, 0.9],
  [0.6, 0.75],
])
// 0 - no buy, 1 - buy

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

const y = tf.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1].map(y => oneHot(y, 2)))
const model = tf.sequential()

model.add(
  tf.layers.dense({
    inputShape: [2],
    units: 3,
    activation: "relu",
  })
)

model.add(
  tf.layers.dense({
    units: 2,
    activation: "softmax",
  })
)

model.compile({
  optimizer: tf.train.adam(0.1),
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
})

async function train() {
  await model.fit(X, y, {
    shuffle: true,
    epochs: 20,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch " + epoch)
        console.log("Loss: " + logs.loss + " accuracy: " + logs.acc)
      },
    },
  })
}
console.log("Training model...")
train().then(() => {
  console.log("Model trained!")
  console.log("Predicting...")
  const prediction = model.predict(tf.tensor2d([[0.4, 0.5]]))
  console.log("Prediction: ", prediction.dataSync())
}).catch(err => {
  console.log("Oh no, there is an error: ", err)
})



