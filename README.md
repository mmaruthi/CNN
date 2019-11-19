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
