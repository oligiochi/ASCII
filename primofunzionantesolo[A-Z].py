# %%

import tensorflow as tf
from keras import layers, models
from keras.utils import to_categorical
import pandas as pd
from keras.regularizers import l1,l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image,ImageOps
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense,MaxPooling2D,Dropout
import os
from random import shuffle
from sklearn.preprocessing import MinMaxScaler

# %%
# Carica il dataset
file = pd.read_csv('DataSet\A_Z Handwritten Data.csv')

# Separa le features (X) e le etichette (y)
X = file.iloc[:, 1:]
y = file.iloc[:, 0]

# Dividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizza i dati
standard_scaler = MinMaxScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

# Reshape dei dati in modo da adattarli alla forma richiesta dai modelli CNN
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Codifica one-hot delle etichette
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%
model = models.Sequential()

# Primo strato convoluzionale
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# Secondo strato convoluzionale
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flattening
model.add(layers.Flatten())

# Primo strato completamente connesso
model.add(layers.Dense(128, activation='relu'))

# Strato di output
model.add(layers.Dense(26, activation='softmax'))

model.summary()

# %%
#compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)

# %%
epoche=10
numero_batch=256
#training
# Training
# Addestramento del modello usando il generatore di immagini con data augmentation
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoche, batch_size=numero_batch)

# %%
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Plot della Loss e dell'Accuracy
plt.figure(figsize=(15, 5))

# Plot della Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot dell'Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(range(epoche), [test_acc] * epoche, label='Test Accuracy', linestyle='--', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Salva il modello
model.save('ASCII_model.h5')


# %%
def plot_predictions_with_probabilities(model, X_test_reshaped, y_test_categorical):
    # Generate predictions
    predictions = model.predict(X_test_reshaped)

    # Set image dimensions
    img_height, img_width = 28, 28
    figsize_height, figsize_width = 3, 15
    tot = 10

    # Create a figure
    fig, axes = plt.subplots(nrows=2, ncols=tot, figsize=(figsize_width, figsize_height))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    # Iterate over predictions
    for i in range(tot):
        # Display the image and set dimensions
        axes[0, i].imshow(np.squeeze(X_test_reshaped[i]), cmap='gray')
        axes[0, i].set_title(f'Predicted: {np.argmax(predictions[i])}\nTrue: {np.argmax(y_test_categorical[i])}')
        axes[0, i].axis('off')

        # Print probabilities for each class
        probabilities = predictions[i]
        for j in range(len(probabilities)):
            axes[1, i].text(0.5, 0.5 - j * 0.1, f"{chr(65 + j)}: {probabilities[j]:.4f}",
                            transform=axes[1, i].transData,
                            color='blue' if j == np.argmax(y_test_categorical[i]) else 'black',
                            fontsize=8, ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'pad': 0})

        # Set cell dimensions
        axes[0, i].set_aspect('equal', adjustable='box')
        axes[0, i].set_xlim([0, img_width])
        axes[0, i].set_ylim([0, img_height])

        # Remove axes for the probability cell
        axes[1, i].axis('off')

    # Remove axes for all cells
    for ax in axes.flat:
        ax.axis('off')

    # Show the figure
    plt.show()

# Execute the code
plot_predictions_with_probabilities(model, X_test, y_test)


# %%
def leggi_numeroimmagine():
    try:
        df = pd.read_json("config.json", orient="records", lines=True)
        numeroimmagine = df["numeroimmagine"].iloc[0]
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError):
        numeroimmagine = 0

    return numeroimmagine
def estrai_parte_disegnata(imagecanvas):
    gray = cv2.cvtColor(imagecanvas, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Trova i limiti dell'area disegnata
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # Ritaglia l'immagine originale solo all'area disegnata
        bordo = 20
        if h > w:
            offset = (h - w) // 2
            drawn_part = imagecanvas[y-bordo:y+h+bordo, x-bordo-offset:x+w+bordo+offset]
        else:
            offset = (w - h) // 2
            drawn_part = imagecanvas[y-bordo-offset:y+h+bordo+offset, x-bordo:x+w+bordo]
    else:
        # Se non ci sono contorni, restituisci un'immagine vuota
        drawn_part = np.zeros_like(imagecanvas)

    return drawn_part


def cam():
    imagecanvas = np.zeros((480, 640, 3), dtype="uint8")
    numeroimmagine = leggi_numeroimmagine()
    x1, y1 = 0, 0
    punte = [4, 8, 12, 16, 20]
    iniziodita = [1, 5, 9, 13, 17]
    detector = mp.solutions.hands
    hands = detector.Hands()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        combined_image = cv2.addWeighted(img, 0.7, imagecanvas, 0.3, 0)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, detector.HAND_CONNECTIONS)
                landmarks = handLms.landmark
                posizione = dita_alzate(punte, iniziodita, landmarks)

                if posizione == [1, 1, 0, 0, 0]:
                    #print("draw mode")
                    cv2.putText(combined_image, "draw mode", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    x, y = int(landmarks[8].x * imagecanvas.shape[1]), int(landmarks[8].y * imagecanvas.shape[0])

                    if x1 == 0 and y1 == 0:
                        x1, y1 = x, y

                    cv2.line(img, (x, y), (x1, y1), (255, 255, 255), 10)
                    cv2.line(imagecanvas, (x, y), (x1, y1), (255, 255, 255), 10)
                    x1, y1 = x, y
                else:
                    x1, y1 = 0, 0
                if posizione == [1, 1, 1, 1, 1]:
                    drawn_part = estrai_parte_disegnata(imagecanvas)
                    if np.count_nonzero(drawn_part) != 0:
                        cv2.imwrite(f"Immagini/parte_disegnata{numeroimmagine}.png", drawn_part)
                        numeroimmagine += 1
                        # Resettare il canvas dopo aver salvato la parte disegnata
                        imagecanvas = np.zeros((480, 640, 3), dtype="uint8")

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Combined", combined_image)
        infoaggiuntive=False
        if infoaggiuntive:
            cv2.imshow("Image", img)
            cv2.imshow("Canvas", imagecanvas)
        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame({"numeroimmagine": [numeroimmagine]})
    df.to_json("config.json", orient="records", lines=True)
def dita_alzate(punte, iniziodita, landmarks):
    dita = []
    for i in range(5):
        if landmarks[0]:
            if i != 0:
                punto0 , punto1 = float(landmarks[0].y), float(landmarks[iniziodita[i]].y)
                if not min(punto0, punto1)<landmarks[punte[i]].y < max(punto0, punto1):
                    dita.append(1)
                else:
                    dita.append(0)
            else:
                punto0 , punto1 = float(landmarks[3].x), float(landmarks[iniziodita[i]].x)
                if not min(punto0, punto1)<landmarks[punte[i]].x < max(punto0, punto1):
                    dita.append(1)
                else:
                    dita.append(0)
    return dita
cam()


# %%
print("Inserisci il numero dell'immagine da predire: ")
numeroimmagine = leggi_numeroimmagine()-1
print(f"Numero immagine: {numeroimmagine}")
# Inserisci il percorso dell'immagine
percorso_immagine = f"Immagini/parte_disegnata{numeroimmagine}.png"
# Carica l'immagine utilizzando la libreria PIL
immagine = Image.open(percorso_immagine).convert('L')  # 'L' indica la modalità scala di grigi

# Ridimensiona l'immagine alle dimensioni attese dal modello (28x28)
immagine = immagine.resize((28, 28))
# Visualizza l'immagine utilizzando matplotlib
plt.imshow(immagine, cmap='gray')
plt.title("Immagine")
plt.axis("off")  # Nasconde gli assi
plt.show()

# Converte l'immagine in un array NumPy e normalizzala
dati_immagine = np.array(immagine) / 255.0

# Aggiungi una dimensione per simulare il batch (1, 28, 28, 1)
dati_immagine_normalizzati = np.expand_dims(dati_immagine, axis=(0, 3))

# Effettua le predizioni
predizioni = model.predict(dati_immagine_normalizzati)

for classe in predizioni:
    probabilita_classi = list(enumerate(classe))
    probabilita_classi.sort(key=lambda x: x[1], reverse=True)
    
    print("Probabilità per ogni classe (in ordine decrescente):")
    for i, (indice_classe, probabilita) in enumerate(probabilita_classi):
        print(f"Classe {chr(65 + indice_classe)}: {probabilita}")


# Le predizioni sono nella forma di probabilità per ogni classe, puoi ottenere la classe predetta usando argmax
classe_predetta = chr(65 + np.argmax(predizioni))
print(f"Classe predetta: {classe_predetta}")

# Salva l'immagine predetta nella directory corrispondente
path = f"Immagini_postWork/{classe_predetta}"
elementi = os.listdir(path)
immagine.save(f"{path}/immagine{len(elementi)}.png")


