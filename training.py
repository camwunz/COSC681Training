from utils import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def main():

    real_eeg_data, labels = load_real_eeg_data()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(real_eeg_data, labels, test_size=0.1)

    # Change this for each model
    model = DeepLearningModel().mlp

    # Train the model - changes depending on scikit-learn or Keras
    fit_scikit(X_train, y_train, model)
    # model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    # Evaluate the model - changes depending on scikit-learn or Keras
    test_accuracy = evaluate_scikit(X_test, y_test, model)
    # test_accuracy = evaluate_nn(X_test, y_test, model)

    print(f'Test accuracy: {test_accuracy}')
    # Dump for future use in the backend api
    pickle.dump(model, open('rt_model', 'wb'))

def fit_scikit(X_train, y_train, model):
    X_train = np.array([np.concatenate(data) for data in X_train])
    model.fit(X_train, y_train)

def evaluate_scikit(X_test, y_test, model):
    """
    Predicts the labels for the test data and calculates the accuracy of the model (scikit model)
    """
    result = [0, 0, 0]
    X_test = np.array([np.concatenate(data) for data in X_test])
    # print(X_test.shape, y_test.shape)
    predicted_labels = []
    for data, _class in zip(X_test, y_test):
        prediction = model.predict([data])[0]
        predicted_labels.append(prediction)
        result[0] += 1
        if prediction == _class:
            result[1] += 1
        else:
            result[2] += 1
    print("Accuracy: ", result[1] / result[0])
    show_confusion(y_test, predicted_labels, "CNN", result[1] / result[0])
    return result[1] / result[0]

def evaluate_nn(X_test, y_test, model):
    """
    Predicts the labels for the test data and calculates the accuracy of the model (Keras model)
    """
    result = [0, 0, 0]
    predicted_labels = []
    for test, _class in zip(X_test, y_test):
        prediction = np.argmax(model(test[None, :]))
        predicted_labels.append(prediction)
        result[0] += 1
        if prediction == _class:
            result[1] += 1
        else:
            result[2] += 1
    print("Accuracy: ", result[1] / result[0])
    show_confusion(y_test, predicted_labels, "ANN", result[1] / result[0])
    return result[1] / result[0]

def show_confusion(y_test, predicted_labels, title="Confusion Matrix", accuracy=None):
    # Create the confusion matrix
    cm = confusion_matrix(y_test, predicted_labels)

    class_labels = [str(i) for i in range(9)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[int(x)+1 for x in class_labels], yticklabels=[int(x)+1 for x in class_labels])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for {}, Accuracy={:.3f}".format(title, accuracy))
    plt.show()

if __name__ == "__main__":
    main()