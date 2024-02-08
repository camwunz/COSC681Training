import tensorflow as tf
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import *

class DeepLearningModel:

    def __init__(self, reset=True):
        self.reset = reset

    @property
    def cnn(self):
        if not self.reset:
            try:
                model = tf.keras.models.load_model('cnn_model')
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model
            except OSError:
                pass
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(sample_duration * sampling_rate, num_channels)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(9, activation='softmax')  # For 9-class classification
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    @property
    def ann(self):
        if not self.reset:
            try:
                model = tf.keras.models.load_model('ann_model')
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model
            except OSError:
                pass
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(sample_duration * sampling_rate, num_channels)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(9, activation='softmax')  # For 9-class classification
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    @property
    def svm(self):
        return svm.SVC(decision_function_shape='ovo')
    
    @property
    def randomforest(self):
        return RandomForestClassifier()
    
    @property
    def lda(self):
        return LinearDiscriminantAnalysis()
    
    @property
    def qda(self):
        return QuadraticDiscriminantAnalysis()
    
    @property
    def mlp(self):
        return MLPClassifier()
    
    @property
    def knn(self):
        return KNeighborsClassifier(n_neighbors=40, weights='distance')
