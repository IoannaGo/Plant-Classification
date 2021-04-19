#Mathima_thema_2020-21 / Gogousou Ioanna 

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

import tensorflow as tf 
import numpy as np
import argparse
import datetime
import pdb
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam , RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint , CSVLogger , LearningRateScheduler


parser = argparse.ArgumentParser(description = 'Plant Classification problem using Tensorflow 2.0')

# Parameters for training 
parser.add_argument(
	'--epochs',
	default = 20 ,
	type= int , 
	dest='num_epochs', 
	help='Number of training epochs')

parser.add_argument(
	'--model',
	default='ResNet50V2',
	type=str,
	dest='model',
	help ='What model to use')

parser.add_argument(
	'--batch_size',
	default=16,
	type=int,
	dest='batch_size',
	help='What batch size to use')

parser.add_argument(
	'--height',
	default=224,
	type=int,
	dest = 'height',
	help='Height of input images, input shape = (height,width,channels)' )


parser.add_argument(
	'--width',
	default=224,
	type=int,
	dest = 'width',
	help='Width of input images, input shape = (height,width,channels)' )

args = parser.parse_args()

# pdb.set_trace()

# Get the current working directory and clear the memory
dir = os.getcwd()
tf.keras.backend.clear_session()

#Train and Test directories
train_dataset_dir = Path(dir + "/Data/Train_data/")
test_dataset_dir = Path(dir + "/Data/Test_data/")

plant_list = os.listdir(train_dataset_dir)
plant_classes= len(plant_list)

#Create generators 
def generators(ds_train_dataset_dir,ds_val_dataset_dir,ds_test_dataset_dir ,height , width , batch_size):

	datagen= tf.keras.preprocessing.image.ImageDataGenerator(
		rescale=1.0/255.0,
		validation_split= 0.2
		)

	test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		rescale=1.0/255.0,
		)

	train_generator =datagen.flow_from_directory(
		directory=Path(dir + "/Data/Train_data/"),
		color_mode='rgb',
		batch_size=batch_size,
		target_size=(height,width),
		shuffle = True,
		seed=None,
		subset='training'
		)

	val_generator =datagen.flow_from_directory(
		directory=Path(dir + "/Data/Train_data/"),
		color_mode='rgb',
		batch_size=batch_size,
		target_size=(height,width),
		shuffle = True,
		seed=None,
		subset='validation'
		)

	test_generator = test_datagen.flow_from_directory(
		directory=Path(dir + "/Data/Test_data/"),
		color_mode='rgb',
		batch_size=1,
		target_size=(height,width),
		shuffle = False,
		seed=None,
		class_mode= None
		)

	return train_generator ,val_generator, test_generator, height , width , batch_size

train_generator,val_generator ,test_generator, height , width , batch_size =generators( train_dataset_dir,train_dataset_dir, test_dataset_dir,args.height,args.width,args.batch_size)


from tensorflow.keras.layers import Dense , Dropout

#ReSNet50V2
def ResNet50V2(height , width , num_classes):
	base = tf.keras.applications.ResNet50V2( 
		include_top = False,
		weights='imagenet',
		input_tensor=None,
		input_shape=(height,width,3),
		pooling= 'avg',
		classes = num_classes)

	name = ResNet50V2.__name__

	x = base.output
	x = tf.keras.layers.Dense(2048, activation = 'relu' , name = 'fc2048')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax', name ='fc'+str(num_classes))(x)
	model = tf.keras.Model (base.input , x , name='resnet50')

	return model , name


#MobileNet
def MobileNet(height , width , num_classes):
	base = tf.keras.applications.MobileNet( 
		include_top = False,
		weights='imagenet',
		input_tensor=None,
		input_shape=(height,width,3),
		pooling= 'avg',
		classes = num_classes,
		)

	name = MobileNet.__name__

	x = base.output
	x = tf.keras.layers.Dense(2048, activation = 'relu' , name = 'fc2048')(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	x = tf.keras.layers.Dense(num_classes,activation='softmax',name ='fc'+str(num_classes))(x)
	model = tf.keras.Model (base.input , x , name='mobilenet')

	return model , name


#InceptionV3 
def InceptionV3 (height, width, num_classes):
	base= tf.keras.applications.InceptionV3(
		include_top= False,
		weights='imagenet',
		input_tensor=None,
		input_shape=(height,width,3),
		pooling='avg',
		classes=num_classes,
		)

	name= InceptionV3.__name__

	x=base.output
	x=tf.keras.layers.Dense(2048,activation='relu' , name='fv2048')(x)
	x= tf.keras.layers.Dropout(0.5)(x)
	x= tf.keras.layers.Dense(num_classes,activation='softmax', name='fc'+ str(num_classes))(x)
	model = tf.keras.Model(base.input , x , name='inceptionV3')

	return model, name


#AlexNet
model= tf.keras.models.Sequential([
    keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation='softmax')      
])
name='alexnet'


if args.model == 'ResNet50V2' :
	model, name = ResNet50V2(height,width,plant_classes)
if args.model == 'MobileNet' :
	model, name = MobileNet(height,width,plant_classes)
if args.model == 'InceptionV3' :
	model, name == InceptionV3 (height,width,plant_classes)

model.summary()

# pdb.set_trace()

#Scheduler
def schedule (epoch):
	if epoch < 3 :
		lr= 0.1
	elif epoch < 5 : 
		lr= 0.05
	elif epoch < 7 : 
		lr= 0.02
	elif epoch < 9 : 
		lr= 0.01
	elif epoch <12 :
		lr= 0.005
	else :
		lr= 0.001

	print ("\nLR at epoch {} = {} \n".format(epoch,lr))
	return lr

lr_scheduler = LearningRateScheduler(schedule)


#Compile the model
model.compile(optimizer='Nadam',
	loss='categorical_crossentropy',
	metrics=['accuracy','top_k_categorical_accuracy'])

model_name= name
check_dir = Path(str(dir) + "/" + "check/" + model_name + "/" + str(plant_classes) + "_classes/")  #checkpoints
check_dir = str(check_dir)

#Training , validation and test images 
train_images = train_generator.n
val_images = val_generator.n
test_images = test_generator.n

try:
  os.makedirs(check_dir)
except:
  FileExistsError

#Step sizes
train_step_size = train_images//train_generator.batch_size
valid_step_size = val_images//val_generator.batch_size
test_step_size = test_images // test_generator.batch_size

filepath = Path(check_dir + "/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5")
filepath = str(filepath)

#Checkpointer
checkpointer = ModelCheckpoint(filepath=filepath,
	verbose=1,
	save_weights_only=False,
	mode='auto',
	save_best_only=True)


log_path = Path(check_dir + "/" + 'hist' + str(args.num_epochs) + '_epochs' + model_name + '.log')
log_path = str(log_path)
csv_logger = CSVLogger(log_path) 


#Train the model
model_train = model.fit_generator(train_generator, epochs=args.num_epochs,
	steps_per_epoch=train_images// batch_size,
	callbacks =[csv_logger, checkpointer , lr_scheduler],
	validation_data = val_generator,
	verbose=1,
	validation_steps = val_images // batch_size)


#Test accuracy 
def calculate_test_accuracy(train_generator, test_generator , model):

	predictions = model.predict_generator(test_generator)
	predicted_class_indices = np.argmax(predictions, axis=1)                    
	true_class_indices = test_generator.classes

	inv_train_generator_indices= {v: k for k, v in train_generator.class_indices.items()}
	inv_test_generator_indices= {v: k for k, v in test_generator.class_indices.items()}

	predicted_classes = [inv_train_generator_indices[c] for c in predicted_class_indices]
	true_classes =[inv_test_generator_indices[c] for c in true_class_indices]

	#pdb.set_trace()

	accuracy = np.sum(np.array(true_classes)==np.array(predicted_classes))/len(predicted_classes)
	print("The test set accuracy calculated is:", accuracy)

	return accuracy


model_dir = "F:/Mathima_thema/checkpoints/MobileNet/100_classes/...."               #40 classes

train_generator, val_generator , test_generator , height , width , batch_size = generators(train_dataset_dir,train_dataset_dir,test_dataset_dir , 224 , 224 , 16)
accuracy = calculate_test_accuracy(train_generator , test_generator, model)


#Plotting figures for accuracy and loss function
acc= model_train.history['accuracy']
val_acc = model_train.history['val_accuracy']
loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs=range(len(acc))

plt.plot(epochs, acc ,'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('train&val_acc.png')

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.title('Training and Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('train&val_loss.png')

plt.legend()
plt.show()
