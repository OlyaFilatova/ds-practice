from utils.loader import load_mnist
from wrapper.mnist_classifier import Algorithm, MnistClassifier

print("creating classifier")
classifier = MnistClassifier(Algorithm.RF.value)

print("loading dataset")
training, test = load_mnist(test_size=10)

print('training')
classifier.train(training[0], training[1])

print('predicting')
res = classifier.predict(test[0])

print('expected')
print(test[1])

print('result')
print(res)
