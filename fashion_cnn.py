import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

class DNN:
    def __init__(self):
        self.LEARNING_RATE=0.001
        self.BATCH_SIZE=50
        self.NUM_CLASSES=10
        self.NUM_EPOCHS=20
        self.INPUT_SHAPE=(28,28,1)
        self.model='conv'
        self.getData()
        self.preprocessing()
        self.trainModel()
        self.outputResult()
        self.getFigure()

    def getData(self):
        (self.X_train, self.y_train),(self.X_test, self.y_test)=fashion_mnist.load_data()
        
    def preprocessing(self):
        if self.model=='dense':
            self.X_train=self.X_train.reshape(-1,28*28)
            self.X_test=self.X_test.reshape(-1,28*28)
        elif self.model=='conv':
            self.X_train=self.X_train.reshape(self.X_train.shape[0],28,28,1)
            self.X_test=self.X_test.reshape(self.X_test.shape[0],28,28,1)
        self.X_train=self.X_train.astype('float32')/255.
        self.X_test=self.X_test.astype('float32')/255.
        self.y_train=to_categorical(self.y_train,self.NUM_CLASSES)
        self.y_test=to_categorical(self.y_test,self.NUM_CLASSES)

    def trainModel(self,model='conv'):
        if model=='dense':
            model=Sequential()
            model.add(Dense(512,input_shape=self.INPUT_SHAPE,activation='relu',name='fc1'))
            model.add(Dense(256,activation='relu',name='fc2'))
            model.add(Dropout(rate=0.5))
            model.add(Dense(self.NUM_CLASSES,activation='softmax',name='output'))
        elif model=='conv':
            model=Sequential()
            model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=self.INPUT_SHAPE,kernel_initializer='he_normal')) 
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(rate=0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            #model.add(Dropout(rate=0.3))
            model.add(Dense(self.NUM_CLASSES,activation='softmax',name='output'))
        optimizer=Adam(learning_rate=self.LEARNING_RATE)   #Adam优化器: 计算每个参数的自适应学习率, 学习率0.001
        model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
        self.model=model

    def outputResult(self):
        model=self.model
        print('DNN Model Summary:')
        print(model.summary())
        history=model.fit(self.X_train,self.y_train,verbose=2,batch_size=self.BATCH_SIZE,epochs=self.NUM_EPOCHS)
        self.train_accuracies=history.history['accuracy']
        self.train_loss=history.history['loss']
        result=model.evaluate(self.X_test,self.y_test)
        print(f'Test loss: {result[0]:.4f}')
        print(f'Test accuracy: {result[1]:.4f}')

    def getFigure(self):
         # 可视化测试集的准确性
        plt.title('Accuracy of Fashion-MNIST over epochs using CNN')
        #plt.ylim(0.0, 1.1)
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.plot(self.train_accuracies, label='train')
        plt.plot(self.train_loss, label='loss')
        plt.legend()
        plt.show()

if __name__=='__main__':
    DNN()