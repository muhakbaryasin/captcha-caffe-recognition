import datetime
import sys
from subprocess import call, check_output
from random import randint
from copy import copy
import os
import caffe
import numpy as np

np.set_printoptions(threshold=np.nan)

def modePakBudiCaptext():
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	captext_list = []
	
	# create captcha text modepakbudi sebanyak 10 kali
	for iter_captext in xrange(385):
		
		for each_letter in alphabet:
			while True:				
				captext = ""
				
				# pick index where each_letter would be in
				i_rand = randint(0, 5)
						
				for letter_id in xrange(6):
					random_letter = alphabet[randint(0, 25)]
					
					if i_rand == letter_id:
						captext += each_letter
					else: captext += random_letter
					
					# add new line after last char
					if letter_id == 5:
						captext += "\n"
				
				# tambahkan ke list. kembar juga gak papa
				if captext not in captext_list:
					captext_list.append(captext)
					break
				else:
					print "captext {} exists".format(captext)
					

	return captext_list

def createNewCaptextList():
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create new captext_list")
	captext_list = modePakBudiCaptext()
	
	# append captext_list to captext_list_file
	with open(captext_list_file, "a") as myfile:
		for each_text in captext_list:
			myfile.write(each_text)

def resetCaptexList():
	# reset list text captcha
	call(["rm", "-rf", captext_list_file])
	call(["touch", captext_list_file])
	call(["rm", "-rf", train_list_file])
	call(["touch", train_list_file])

def cleanUpDir(dir_path):
	call(["rm", "-rf", dir_path])
	call(["mkdir", dir_path])
	#call(["touch", dir_path+"supayakeadd"])

def cleanUpLastTrainDir():
	# reset directory train image captcha
	cleanUpDir(train_files_dir)

def cleanUpLastTestDir():
	# reset directory test image captcha
	cleanUpDir(test_files_dir)

def cleanUpRecognizedDir():
	cleanUpDir(recognized_files_dir)

def migrateTestFilesToRecognized(filename):
	call(["mv", test_files_dir+filename, recognized_files_dir+filename])

def convertTestDirToTrainDir():
	call["mv", test_files_dir, train_files_dir]
	

def configNetCaffe():
	params = {}
	params["net"] = "network_captchas_with_3_convolutional_layers_train.prototxt"
	
	# The base learning rate, momentum and the weight decay of the network.
	params["base_lr"] = 0.01
	
	params["momentum"] = 0.9
	params["weight_decay"] = 0.0005
	
	# The learning rate policy
	params["lr_policy"] = "inv"
	params["gamma"] = 0.0001
	params["power"] = 0.75
	
	# Display every params["display"] iterations 
	params["display"] = 500
	
	# The maximum number of iterations
	params["max_iter"] = 20000
	
	# snapshot intermediate results
	params["snapshot"] = 5000
	
	params["snapshot_prefix"] = "temp/snapshots/modepakbudi"
	
	# solver mode: CPU or GPU
	params["solver_mode"] = "GPU"
	
	return params

def getLastSnapshot():
	output = check_output(["ls", "-1t", snapshots_dir])
	output = output.split('\n')
	last_snapshot = {}
	
	for each_file in output:
		if each_file.find("solverstate") > -1:
			last_snapshot['solverstate'] = copy(each_file)
		elif each_file.find("caffemodel") > -1:
			last_snapshot['caffemodel'] = copy(each_file)
		
		if "solverstate" in last_snapshot and "caffemodel" in last_snapshot:
			return last_snapshot

def classifyImage(input_image, network, caffe_model):
	classifier = caffe.Classifier(network, caffe_model, mean=None)
	input_images = [input_image]
	
	prediction = classifier.predict(input_images, oversample=False)
	return prediction

# This function is the inverse function of convertCharacterToClass
def convertClassToCharacter(predictedClass):
	if predictedClass < 10:
		predictedCharacter = chr(predictedClass+48)
		#print 'Predicted digit:', predictedCharacter
	elif predictedClass <= 36:
		predictedCharacter = chr(predictedClass+55)
		#print "Predicted big letter", predictedCharacter
	else:
		predictedCharacter = chr(predictedClass+60)
		#print "Predicted small letter", predictedCharacter
	return predictedCharacter
	

def main():
	"""old_stdout = sys.stdout
	log_file = open("message.log","w")
	sys.stdout = log_file"""
	
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " clean up first")
	resetCaptexList()	
	cleanUpLastTrainDir()
	cleanUpLastTestDir()
	
	# init list dataset train
	print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " init captext list")
	createNewCaptextList()
	
	# active learning loop
	iter_num = 1
	
	# allright_times -> when all item of captext_list_file fully recognized
	all_correct_times = 0
	
	while iter_num < 3 and all_correct_times < 2:
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " iteration number {}".format(iter_num))
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ==========================================")
		
		last_snapshot = getLastSnapshot()
		
		if last_snapshot is None:
			print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create TRAIN image set based on captext list")
			call(["php-cgi", "captcha.php", "dest-directory="+train_files_dir, "captext-list="+captext_list_file])
		
		else:
			# test
			# create test images
			print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create TEST image set based on captext list")
			call(["php-cgi", "captcha.php", "dest-directory="+test_files_dir, "captext-list="+captext_list_file])
			
			all_correct = True
		
			for each_file in os.listdir(test_files_dir):
				image_file_path = test_files_dir + each_file
				correct_string = os.path.splitext(each_file)[0]
				input_image = caffe.io.load_image(image_file_path, color=False)
				prediction = classifyImage(input_image, "network_captchas_with_3_convolutional_layers.prototxt", snapshots_dir+last_snapshot['caffemodel'])
				
				predictedString = ""
				numberOfDigits = 6
				classesPerDigit = 63
				
				for x in xrange(0, numberOfDigits):
					predictedChar = prediction[0][63*x:63*(x+1)]
					predictedChar = predictedChar * sum(predictedChar) ** -1
					predictedClass = predictedChar.argmax()
					predictedCharacter = convertClassToCharacter(predictedClass)
					predictedString+=predictedCharacter
				
				if predictedString != correct_string:
					# anggap bagian ini berarti dy masih salah baca
					all_correct = False
					pass
				else:
					# anggap bagian ini berarti dy udah bisa baca
					migrateTestFilesToRecognized(each_file)
					pass
			
			if all_correct:
				print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " wow all correct. ")
				all_correct_times += 1
				createNewCaptextList()
			else :
				convertTestDirToTrainDir()
		
		
		# reset images db
		call(["rm", "-rf", train_db_dir])
		
		# create train list filename and label
		call(["bash", "create-train-list.sh", train_files_dir, train_list_file])
		
		# create new images db
		call(["convert_imageset", "--backend=leveldb","--gray", "--resize_height=0", "--resize_width=0", "--shuffle=true", train_files_dir, train_list_file, train_db_dir])
		cleanUpLastTrainDir()
		
		last_snapshot = getLastSnapshot()
		
		print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " caffe TRAIN")
		# caffe train
		if last_snapshot is None:
			print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " no snapshot")
			call(["caffe", "train", "--solver=captcha_solver.prototxt"])
		else:
			print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " use snapshot " + snapshots_dir+last_snapshot['solverstate'])
			call(["caffe", "train", "--solver=captcha_solver.prototxt", "--snapshot="+snapshots_dir+last_snapshot['solverstate'] ])
		
		
		iter_num += 1
		break
	
	#sys.stdout = old_stdout
	#log_file.close()
		
		
	
if __name__ == "__main__":
	train_files_dir = "temp/train-files/"
	test_files_dir = "temp/test-files/"
	recognized_files_dir = "temp/recognized-files/"
	captext_list_file = "temp/captcha-text-list.txt"
	train_list_file = "temp/train-list.txt"
	train_db_dir = "temp/train.db"
	snapshots_dir = "temp/snapshots/"
	main()
