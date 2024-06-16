# %%
import tensorflow as tf
from keras import backend as K
from keras import layers, models
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import os
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time


# %%
# Carica il dataset
file = pd.read_csv('DataSet\A_Z Handwritten Data.csv')
# Separa le features (X) e le etichette (y)
X = file.iloc[:, 1:]
y = file.iloc[:, 0]

# Dividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sovracampionamento delle classi minoritarie sul set di addestramento
#print("Sovracampionamento delle classi minoritarie")
#smote = SMOTE(random_state=42)
#X_train, y_train = smote.fit_resample(X_train, y_train)
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
f=X[y == 8]
random_int = random.randint(0, f.shape[0])
plt.imshow(f.iloc[random_int].values.reshape(28, 28), cmap='gray')

# %%
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
dataset_alphabets = file.copy()
dataset_alphabets.rename(columns={'0':'label'}, inplace=True)
dataset_alphabets['label'] = dataset_alphabets['label'].map(alphabets_mapper)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']
dataset_alphabets = dataset_alphabets.groupby('label')
label_size=dataset_alphabets.size()
# Calcola la nuova distribuzione delle classi dopo il sovracampionamento
y_bo=np.argmax(y_train, axis=1)
resampled_class_distribution = pd.Series(y_bo).value_counts().sort_index()
#label_size=label_size.sort_index(ascending=False)
my_palette = sns.color_palette("coolwarm", len(label_size))
sns.barplot(y=label_size, x=class_labels, palette=colors_from_values(label_size, "YlOrRd"))
sns.barplot(y=resampled_class_distribution.values, x=class_labels, palette=colors_from_values(label_size, "YlOrRd"))
#label_size.plot.barh(figsize=(10,10), color=my_palette)
plt.show()
# Calcola il peso di ciascuna lettera
class_weights = {}
for letter, frequency in label_size.items():
    class_weights[ord(letter)-ord('A')] = 1 / frequency

# Normalizza i pesi
total_weight = sum(class_weights.values())
for letter, weight in class_weights.items():
  class_weights[letter] = weight / total_weight

# Stampa i pesi di classe
print(class_weights)



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
from pathlib import Path
from time import strftime

def get_log(root="ASCII/Logs"):
    return Path(root) / strftime("run %Y-%m-%d %H-%M-%S")
run_log=get_log()
tensorboard=tf.keras.callbacks.TensorBoard(run_log,profile_batch=(100,200))

# %%
epoche=10
numero_batch=256
#training
# Training
# Addestramento del modello usando il generatore di immagini con data augmentation
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoche, batch_size=numero_batch,callbacks=[tensorboard])

# %% [markdown]
# # Studi Post Addestramento

# %%
%load_ext tensorboard
print(run_log)
%tensorboard --logdir=./ASCII/Logs

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
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
#plt.plot(range(epoche), [test_acc] * epoche, label='Test Accuracy', linestyle='--', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Salva il modello
model.save('ASCII_model.h5')


# %% [markdown]
# ## Predizioni

# %%
def plot_predictions_with_probabilities(model, X_test_reshaped, y_test_categorical):
    # Genera le predizioni
    predictions = model.predict(X_test_reshaped)
    predicted_labels = np.argmax(predictions, axis=1)
    y_test_true_labels = np.argmax(y_test_categorical, axis=1)  # Etichette vere

    # Dimensioni dell'immagine
    img_height, img_width = 28, 28
    figsize_height, figsize_width = 3, 15
    tot = 10

    # Crea una figura
    fig, axes = plt.subplots(nrows=2, ncols=tot, figsize=(figsize_width, figsize_height))
    fig.subplots_adjust(wspace=0.05, hspace=0.1)

    # Itera sulle predizioni
    for i in range(tot):
        # Visualizza l'immagine e imposta le dimensioni
        axes[0, i].imshow(np.squeeze(X_test_reshaped[i]), cmap='gray')
        axes[0, i].set_title(f'Predetto: {np.argmax(predictions[i])}\nVero: {y_test_true_labels[i]}')
        axes[0, i].axis('off')

        # Stampa le probabilità per ciascuna classe
        probabilities = predictions[i]
        for j in range(len(probabilities)):
            color = 'black'
            if j == y_test_true_labels[i]:
                color = 'blue'  # Colora in blu la classe vera
            elif j == np.argmax(probabilities):
                color = 'red'  # Colora in rosso la probabilità massima predetta

            axes[1, i].text(0.5, 0.5 - j * 0.1, f"{chr(65 + j)}: {probabilities[j]:.4f}",
                            transform=axes[1, i].transData,
                            color=color,
                            fontsize=8, ha='center', bbox={'facecolor': 'white', 'edgecolor': 'white', 'pad': 0})

        # Imposta le dimensioni delle celle
        axes[0, i].set_aspect('equal', adjustable='box')
        axes[0, i].set_xlim([0, img_width])
        axes[0, i].set_ylim([0, img_height])

        # Rimuovi gli assi per la cella della probabilità
        axes[1, i].axis('off')

    # Rimuovi gli assi per tutte le celle
    for ax in axes.flat:
        ax.axis('off')
        
    class_labels = [chr(65 + i) for i in range(26)]  # Genera le etichette delle classi (A-Z)

    # Mostra la figura
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test_true_labels, predicted_labels)
    m = 2.5
    cmap = sns.cubehelix_palette(light=1, as_cmap=True, hue=1, dark=0.2)
    plt.figure(figsize=(8 * m, 6 * m))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Vere')
    plt.title('Confusion Matrix')
    plt.show()

# Esegui il codice
plot_predictions_with_probabilities(model,  X_test, y_test)

# %% [markdown]
# # Interfaccia Da Telecamera e Riconoscimento delle Lettere

# %%
if not ('model' in locals() or 'model' in globals()):
    print("Caricamento del modello")
    model = tf.keras.models.load_model('ASCII_model.h5')

cancella = False
alphabets_mapper = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

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
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawn_parts = []
    drawn_part_normalizeds = []
    if contours:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] == -1:
                x, y, w, h = cv2.boundingRect(contour)
                bordo = 20
                if h > w:
                    offset = (h - w) // 2
                    drawn_part = imagecanvas[y - bordo:y + h + bordo, x - bordo - offset:x + w + bordo + offset]
                else:
                    offset = (w - h) // 2
                    drawn_part = imagecanvas[y - bordo - offset:y + h + bordo + offset, x - bordo:x + w + bordo]
                drawn_parts.append((drawn_part, (x, y, w, h)))
                drawn_part_resized = cv2.resize(drawn_part, (28, 28))
                drawn_part_recolor = cv2.cvtColor(drawn_part_resized, cv2.COLOR_BGR2GRAY)
                drawn_part_normalized = drawn_part_recolor / 255.0
                drawn_part_normalized = np.expand_dims(drawn_part_normalized, axis=(0, 3))
                drawn_part_normalizeds.append(drawn_part_normalized)
        drawn_part_normalizeds = np.concatenate(drawn_part_normalizeds, axis=0)
    return drawn_parts, drawn_part_normalizeds

def dita_alzate(punte, iniziodita, landmarks):
    dita = []
    for i in range(5):
        if landmarks[0] and landmarks[iniziodita[i]] and landmarks[punte[i]]:
            if i != 0:
                punto0, punto1 = float(landmarks[0].y), float(landmarks[iniziodita[i]].y)
                if not min(punto0, punto1) < landmarks[punte[i]].y < max(punto0, punto1):
                    dita.append(1)
                else:
                    dita.append(0)
            else:
                punto0, punto1 = float(landmarks[3].x), float(landmarks[iniziodita[i]].x)
                if not min(punto0, punto1) < landmarks[punte[i]].x < max(punto0, punto1):
                    dita.append(1)
                else:
                    dita.append(0)
        else:
            dita.append(0)
    return dita

def disegna_bounding_box(image, drawn_parts, classi_predette):
    for i, (drawn_part, (x, y, w, h)) in enumerate(drawn_parts):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if classi_predette and i < len(classi_predette):
            cv2.putText(image, f"{classi_predette[i]}", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def esegui_previsioni(model, drawn_part_normalizeds):
    predizioni = model.predict(drawn_part_normalizeds)
    keys = np.argmax(predizioni, axis=1)
    mapped_values = [alphabets_mapper[key] for key in keys]
    return mapped_values

def process_canvas(imagecanvas, model, queue):
    global cancella
    if not cancella:
        drawn_parts, drawn_part_normalizeds = estrai_parte_disegnata(imagecanvas)
        classi_predette = esegui_previsioni(model, drawn_part_normalizeds)
        queue.put((drawn_parts, classi_predette))

def cam():
    global cancella
    global classi_predette
    imagecanvas = np.zeros((480, 640, 3), dtype="uint8")
    numeroimmagine = leggi_numeroimmagine()
    x1, y1 = 0, 0
    punte = [4, 8, 12, 16, 20]
    iniziodita = [1, 5, 9, 13, 17]
    detector = mp.solutions.hands
    hands = detector.Hands()
    mpDraw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    classi_predette = []
    executor = ThreadPoolExecutor(max_workers=1)
    frame_count = 0
    queue = Queue()
    start_time = time.time()
    draw_mod_on = False
    future,box, classi = None, None,None

    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        combined_image = cv2.addWeighted(img, 0.7, imagecanvas, 0.3, 0)

        current_time = time.time()
        fps = frame_count / (current_time - start_time) if current_time - start_time > 0 else 0
        cv2.putText(combined_image, f"FPS: {int(fps)}", (combined_image.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, detector.HAND_CONNECTIONS)
                landmarks = handLms.landmark
                posizione = dita_alzate(punte, iniziodita, landmarks)

                if posizione == [0, 1, 0, 0, 1] or posizione == [1, 1, 0, 0, 1]:
                    draw_mod_on = True
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
                    print("Salvataggio immagine")
                    cancella = True
                    drawn_parts, drawn_part_normalizeds = estrai_parte_disegnata(imagecanvas)
                    if drawn_parts:
                        for i, (drawn_part, (x, y, w, h)) in enumerate(drawn_parts):
                            if np.count_nonzero(drawn_part) != 0:
                                cv2.imwrite(f"Immagini/parte_disegnata{numeroimmagine}.png", drawn_part)
                                numeroimmagine += 1
                    imagecanvas = np.zeros((480, 640, 3), dtype="uint8")

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 0:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        if draw_mod_on and np.count_nonzero(imagecanvas) > 0:
            future=executor.submit(process_canvas, imagecanvas, model, queue)
            draw_mod_on = False

        if not queue.empty():
            box, classi = queue.get()
            queue.queue.clear()

        if (box is not None and classi is not None) or cancella:
            if cancella:
                classi = []
                box = []
                queue.queue.clear()
            disegna_bounding_box(combined_image, box, classi)
        
        if cancella and future and future.done():
            cancella = False

        if combined_image is not None and combined_image.shape[0] > 0 and combined_image.shape[1] > 0:
            cv2.imshow("Combined", combined_image)
        else:
            print("Combined image is invalid or empty.")
        key = cv2.waitKey(1)
        #print(cancella)
        if key == ord('q') or key == 27:
            break
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame({"numeroimmagine": [numeroimmagine]})
    df.to_json("config.json", orient="records", lines=True)

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
if not 'model' in locals() or 'model' in globals():
    print("Caricamento del modello")
    model=tf.keras.models.load_model('ASCII_model.h5')
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



