Approach for the model :

Target is to acheive 99.4% accuracy within 15K parameters . No FCs . No bias.

1. Input is MNIST data which is 28 * 28 pixels . As it's not complex , i felt it's not needed to use many filters. 
   I could see more filters are causing an increase in params. So, I confined myself to maximum 16 filters as i felt it's
   enough to extract features.
   
2. After adding two layers with 16 filters each , reduced it to 24 . Then used 1 *1 to downsize to 10 filters 
3. Afterward , added a max pool layer to bring to 12 * 12 and added multiple layers again until i got 1 *1 and then flattened 
4. Used- Batch Normalization and Drop out regularization techniques and every step to accelerate the training process and 
   improve performance.
 5. Tuned learning rate to acheive 99.4% accuracy
 6. Removed bias using bias = false
 
 Key things :
    - confined to 16 filters 
    - Used regularization techniques - Batch normalization and drop out 
    - Removed bias . use_bias=False 
    - Tuned learning rate to improve accuracy 


Model for reference :
=============================
model = Sequential()
 
model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu', input_shape=(28,28,1))) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu')) #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, 1, use_bias=False, activation='relu')) #24

model.add(MaxPooling2D(pool_size=(2, 2)))#12

model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu'))#10
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu'))#8
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu'))#6
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, use_bias=False, activation='relu'))#4
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 4, 4, use_bias=False)) #1
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Activation('softmax'))



Log data :
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 9s 156us/step - loss: 0.1131 - acc: 0.9519 - val_loss: 0.0310 - val_acc: 0.9908
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 6s 103us/step - loss: 0.1054 - acc: 0.9533 - val_loss: 0.0357 - val_acc: 0.9891
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0982 - acc: 0.9552 - val_loss: 0.0225 - val_acc: 0.9929
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 6s 103us/step - loss: 0.0968 - acc: 0.9548 - val_loss: 0.0244 - val_acc: 0.9934
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0895 - acc: 0.9581 - val_loss: 0.0240 - val_acc: 0.9935
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0900 - acc: 0.9568 - val_loss: 0.0215 - val_acc: 0.9938
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0884 - acc: 0.9586 - val_loss: 0.0208 - val_acc: 0.9941
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0866 - acc: 0.9582 - val_loss: 0.0194 - val_acc: 0.9948
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 6s 103us/step - loss: 0.0844 - acc: 0.9597 - val_loss: 0.0209 - val_acc: 0.9941
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 6s 103us/step - loss: 0.0851 - acc: 0.9583 - val_loss: 0.0203 - val_acc: 0.9950
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 6s 103us/step - loss: 0.0832 - acc: 0.9585 - val_loss: 0.0200 - val_acc: 0.9944
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0824 - acc: 0.9595 - val_loss: 0.0214 - val_acc: 0.9943
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0818 - acc: 0.9595 - val_loss: 0.0174 - val_acc: 0.9953
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 6s 103us/step - loss: 0.0820 - acc: 0.9598 - val_loss: 0.0189 - val_acc: 0.9949
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0812 - acc: 0.9592 - val_loss: 0.0180 - val_acc: 0.9948
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0796 - acc: 0.9606 - val_loss: 0.0208 - val_acc: 0.9939
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 6s 100us/step - loss: 0.0820 - acc: 0.9590 - val_loss: 0.0220 - val_acc: 0.9940
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0786 - acc: 0.9601 - val_loss: 0.0202 - val_acc: 0.9947
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0788 - acc: 0.9614 - val_loss: 0.0197 - val_acc: 0.9949
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0795 - acc: 0.9598 - val_loss: 0.0205 - val_acc: 0.9943


Score of the Model :
===================
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
[0.02046401365385973, 0.9943]
