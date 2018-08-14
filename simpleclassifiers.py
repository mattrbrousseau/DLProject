import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


config = tf.ConfigProto()
config.intra_op_parallelism_threads = 16
config.inter_op_parallelism_threads = 16
sess = tf.Session(config=config)

classes = ["Tearing","Bus","Shatter","Gunshot, gunfire","Fireworks","Writing",\
		"Computer keyboard","Scissors","Microwave oven","Keys jangling",\
		"Drawer open or close","Squeak","Knock","Telephone","Saxophone",\
		"Oboe","Flute","Clarinet","Acoustic guitar","Tambourine","Glockenspiel",\
		"Gong","Snare drum","Bass drum","Hi-hat","Electric piano","Harmonica",\
		"Trumpet","Violin, fiddle","Double bass","Cello","Chime","Cough",\
		"Laughter","Applause","Finger snapping","Fart","Burping, eructation",\
		"Cowbell","Bark","Meow"]

def model_3hLayers(features,labels,obs,basepath):
	'''
	creating model
	'''

	model = Sequential()
	model.add(Dense(units=2560, activation='relu', input_dim=52000))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dense(units=41, activation='softmax'))

	model.compile(loss=tf.keras.losses.categorical_crossentropy,
	              optimizer=tf.keras.optimizers.Adam())

	#DEFAULT PARAMETERS FOR Adam:
	#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False

	from keras.utils.np_utils import to_categorical

	for i in range(len(labels)):
		
		if labels[i] == "Computer_keyboard":
			labels[i] = "Computer keyboard"
		if labels[i] == "Microwave_oven":
			labels[i] = "Microwave oven"
		if labels[i] == "Keys_jangling":
			labels[i] = "Keys jangling"
		if labels[i] == "Acoustic_guitar":
			labels[i] = "Acoustic guitar"
		if labels[i] == "Gunshot_or_gunfire":
			labels[i] = "Gunshot, gunfire"
		if labels[i] == "Snare_drum":
			labels[i] = "Snare drum"
		if labels[i] == "Microwave_oven":
			labels[i] = "Microwave oven"
		if labels[i] == "Bass_drum":
			labels[i] = "Bass drum"
		if labels[i] == "Electric_piano":
			labels[i] = "Electric piano"
		if labels[i] == "Double_bass":
			labels[i] = "Double bass"
		if labels[i] == "Finger_snapping":
			labels[i] = "Finger snapping"
		if labels[i] == "Burping_or_eructation":
			labels[i] = "Burping, eructation"
		if labels[i] == "Drawer_open_or_close":
			labels[i] = "Drawer open or close"
		if labels[i] == "Violin_or_fiddle":
			labels[i] = "Violin, fiddle"
		labels[i].replace("_"," ")

	labelsHotVector = to_categorical([classes.index(x) for x in labels],num_classes=41)

	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(features,labelsHotVector,test_size=0.1, random_state=2018, shuffle = True)

	'''
	Training
	'''
	print('**training the model**')
	model.fit([x_train], [y_train], epochs=5, batch_size=50,verbose=2)

	# Save weights to a TensorFlow Checkpoint file
	model.save_weights('./model_'+obs)

	'''
	Testing
	'''
	print('**testing the model**')

	testing(x_test,y_test,model,obs,basepath)

	print("End")


def testing(x_test,y_test,model,obs,basepath):
	print('**quality measures**')
	y_pred = model.predict([x_test], batch_size=100)
	accuracy = tf.keras.metrics.categorical_accuracy(y_test, y_pred)
	#loss_and_metrics = model.evaluate(x_test, y_test, batch_size=100)
	file4results = open(basepath+'results_'+obs,'w')
	file4results.write(str(sess.run(accuracy)))
	file4results.close()

def model_14hLayers(features,labels,obs,basepath):
	'''
	creating model
	'''
	print(features)

	model = Sequential()
	model.add(Dense(units=5000, activation='relu', input_dim=52000))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=2560, activation='relu'))
	model.add(Dense(units=1024, activation='relu'))
	model.add(Dense(units=41, activation='softmax'))

	model.compile(loss=tf.keras.losses.categorical_crossentropy,
	              optimizer=tf.keras.optimizers.Adam())

	#DEFAULT PARAMETERS FOR Adam:
	#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False

	from keras.utils.np_utils import to_categorical

	for i in range(len(labels)):
		
		if labels[i] == "Computer_keyboard":
			labels[i] = "Computer keyboard"
		if labels[i] == "Microwave_oven":
			labels[i] = "Microwave oven"
		if labels[i] == "Keys_jangling":
			labels[i] = "Keys jangling"
		if labels[i] == "Acoustic_guitar":
			labels[i] = "Acoustic guitar"
		if labels[i] == "Gunshot_or_gunfire":
			labels[i] = "Gunshot, gunfire"
		if labels[i] == "Snare_drum":
			labels[i] = "Snare drum"
		if labels[i] == "Microwave_oven":
			labels[i] = "Microwave oven"
		if labels[i] == "Bass_drum":
			labels[i] = "Bass drum"
		if labels[i] == "Electric_piano":
			labels[i] = "Electric piano"
		if labels[i] == "Double_bass":
			labels[i] = "Double bass"
		if labels[i] == "Finger_snapping":
			labels[i] = "Finger snapping"
		if labels[i] == "Burping_or_eructation":
			labels[i] = "Burping, eructation"
		if labels[i] == "Drawer_open_or_close":
			labels[i] = "Drawer open or close"
		if labels[i] == "Violin_or_fiddle":
			labels[i] = "Violin, fiddle"
		labels[i].replace("_"," ")

	labelsHotVector = to_categorical([classes.index(x) for x in labels],num_classes=41)

	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(features,labelsHotVector,test_size=0.1, random_state=2018, shuffle = True)

	'''
	Training
	'''
	print('**training the model**')
	model.fit([x_train], [y_train], epochs=5, batch_size=50,verbose=2)

	# Save weights to a TensorFlow Checkpoint file
	model.save_weights('./model_'+obs)

	'''
	Testing
	'''
	print('**testing the model**')

	testing(x_test,y_test,model,obs,basepath)

	print("End")
