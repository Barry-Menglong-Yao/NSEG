import matplotlib.pyplot as plt
import numpy as np 
def a():
    # x = np.arange(0, 5, 0.1)
    # y = np.sin(x)
    cat = ["bored", "happy", "bored", "bored", "happy", "bored"]
    dog = ["happy", "happy", "happy", "happy", "bored", "bored"]
    activity = ["combing", "drinking", "feeding", "napping", "playing", "washing"]

    fig, ax = plt.subplots()
    ax.plot(activity, dog, label="dog")
    ax.plot(activity, cat, label="cat")
    ax.legend()

    plt.show()
    
a()

def b():

    history={"accuracy":[],"loss":[]}
    history["accuracy"].append(1)
    history["accuracy"].append(5)
    history["accuracy"].append(9)
    history["val_accuracy"].append(1)
    history["val_accuracy"].append(5)
    history["val_accuracy"].append(9)
    history["loss"].append(11)
    history["loss"].append(15)
    history["loss"].append(41)

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



