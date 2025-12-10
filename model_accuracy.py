import keras
from keras.saving.experimental.saving_lib import load_model

keras.models.load_model('Model/keras_model.h5')
model = load_model('keras_model.h5')
model.evaluate()

