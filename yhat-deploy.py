cd ~/workspace/github.com/yhat/demo-image-recognizer

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import pprint as pp
import matplotlib.pyplot as plt

# http://keras.io/
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

plt.imshow(cv2.imread('images/pig.jpg'))


labels = [" ".join(row.split(' ')[1:]) for row in open("./labels.txt").read().strip().split('\n')]
df_labels = pd.DataFrame({"label": labels})
im = cv2.resize(cv2.imread('images/pig.jpg'), (224, 224)).astype(np.float32)
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)


# Test pretrained model
model = VGG_16('data/vgg16_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(im)
pred = dict(zip(labels, model.predict_proba(im)[0]))
best_guess = labels[np.argmax(out)]

output = []
guesses = np.array(labels)[np.argsort(out[0])].tolist()
guesses.reverse()
for item in guesses[:10]:
    output.append(item)

output = ", ".join(output)
best_guess = ",".join(output.split(", ")[:10])

print "It's a %s" % best_guess
{ "guess": best_guess, "prob": pred }


from yhat import Yhat, YhatModel, preprocess
from PIL import Image
from StringIO import StringIO
import base64

class ImageRecognizer(YhatModel):
    REQUIREMENTS = [
        "opencv"
    ]
    @preprocess(in_type=dict, out_type=dict)
    def execute(self, data):
        img64 = data['image64']
        binaryimg = base64.decodestring(img64)
        pilImage = Image.open(StringIO(binaryimg))
        image = np.array(pilImage)
        resized_image = cv2.resize(image, (224, 224)).astype(np.float32)
        resized_image[:,:,0] -= 103.939
        resized_image[:,:,1] -= 116.779
        resized_image[:,:,2] -= 123.68
        resized_image = resized_image.transpose((2,0,1))
        resized_image = np.expand_dims(resized_image, axis=0)

        out = model.predict(resized_image)
        # pred = dict(zip(labels, model.predict_proba(im)[0]))

        output = []
        guesses = np.array(labels)[np.argsort(out[0])].tolist()
        guesses.reverse()
        for item in guesses[:10]:
            output.append(item)

        output = ", ".join(output)
        guesses = ",".join(output.split(", ")[:5])
        print "It's a %s" % guesses
        return { "guess": guesses }


testdata = {
    "image64": open('./test-image.base64', 'rb').read()
}
print ImageRecognizer().execute(testdata)

yh = Yhat(USERNAME, APIKEY, URL)
yh.deploy("ImageRecognizer", ImageRecognizer, globals(), True)
