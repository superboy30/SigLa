from start import *


DATA_PATH = os.path.join('Collect') 

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
X.shape
y = to_categorical(labels).astype(int)

class CustomSaver(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and epoch % 100 == 0:  # Save epochs every hundred 
            self.model.save("action_{}.h5".format(epoch))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
saver = CustomSaver()
# model_checkpoint = ModelCheckpoint(
#     filepath='model.h5',
#     save_weights_only=True,
#     monitor='categorical_accuracy',
#     mode='max',
#     save_best_only=True
# )

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback, saver])

# 1k, 500, 300, 200

model.save('action.h5')