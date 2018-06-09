import pdb
import datetime
import sys
from subprocess import call, check_output
from random import randint
from copy import copy
import os
import caffe
import numpy as np
from math import ceil
from classify import *
from time import sleep

np.set_printoptions(threshold=np.nan)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('iterate.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

os.environ['GLOG_minloglevel'] = '2'

def modePakBudiCaptext(total_num_text=10000):
	total_num_text = float(total_num_text)
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	captext_list = []
	captext_len = 6 # len of captcha text
	
	iteration_num = int(ceil(total_num_text / len(alphabet) ))
	
	# create captcha text modepakbudi sebanyak iteration_num untuk memenuhi kuota dataset
	for iter_captext in xrange(iteration_num):
		
		for each_letter in alphabet:
			while True:				
				captext = ""
				
				# pick index where each_letter would be in
				i_rand = randint(0, captext_len - 1)
						
				for letter_id in xrange(6):
					random_letter = alphabet[randint(0, len(alphabet) - 1 )]
					
					if i_rand == letter_id:
						captext += each_letter
					else: captext += random_letter
					
					# add new line after last char
					#if letter_id == 5:
					#	captext += "\n"
				
				# tambahkan ke list. kembar juga gak papa
				if captext not in captext_list:
					captext_list.append(captext)
					break
				else:
					print "captext {} exists".format(captext)
					

	return captext_list

def subcall(list_):
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + (" ".join(list_)))
	call(list_)

def createNewCaptextList():
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create new captext_list")
	captext_list = modePakBudiCaptext()
	subcall(["rm", "-rf", captext_list_file])
	subcall(["touch", captext_list_file])
	
	# append captext_list to captext_list_file
	for each_text in captext_list:
		call("echo {} >> {}".format(each_text, captext_list_file), shell="True")
		
	#with open(captext_list_file, "a") as myfile:
	#	for each_text in captext_list:
	#		myfile.write(each_text)

def resetCaptexList():
	# reset list text captcha
	subcall(["rm", "-rf", captext_list_file])
	subcall(["touch", captext_list_file])
	subcall(["rm", "-rf", train_list_file])
	subcall(["touch", train_list_file])

def cleanUpDir(dir_path):
	subcall(["rm", "-rf", dir_path])
	subcall(["mkdir", dir_path])
	#subcall(["touch", dir_path+"supayakeadd"])

def cleanUpLastTrainDir():
	# reset directory train image captcha
	cleanUpDir(train_files_dir)

def cleanUpLastTestDir():
	# reset directory test image captcha
	cleanUpDir(test_files_dir)

def cleanUpRecognizedDir():
	cleanUpDir(recognized_files_dir)

def migrateTestFilesToRecognized(filename, post_fix=None):
	if post_fix is not None:
		post_fix = "__"+str(post_fix)
	
	call(["mv", test_files_dir+filename, recognized_files_dir+filename+post_fix])

def convertTestDirToTrainDir():
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " Make TEST image set become TRAIN image set")
	subcall(["rm", "-rf", train_files_dir])
	subcall(["mv", test_files_dir, train_files_dir])
	subcall(["mkdir", test_files_dir])	

def configNewNetCaffe():
	subcall(["rm", "-rf", caffe_config_file])
	subcall(["touch", caffe_config_file])
	
	with open(caffe_config_file, "a") as myfile:
		for each_idx in caffe_config:
			myfile.write("{}: {}\n".format(each_idx, caffe_config[each_idx]))
	
	#for idx in caffe_config:		
	#	subcall("echo {}: {} >> {}".format(each_idx, caffe_config[each_idx], caffe_config_file), shell="True")
	

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
			real_prefix = caffe_config['snapshot_prefix'].split("/")[-1].replace("\"", "")+ "_iter_"
			last_snapshot['iteration'] = last_snapshot['solverstate'].replace(real_prefix,"").replace(".solverstate","")
			
			return last_snapshot

#def classifyImage(input_image, network, caffe_model):
#	classifier = caffe.Classifier(network, caffe_model, mean=None)
#	input_images = [input_image]
#	
#	prediction = classifier.predict(input_images, oversample=False)
#	return prediction

def main():
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " clean up first")
	resetCaptexList()	
	cleanUpLastTrainDir()
	cleanUpLastTestDir()
	
	# correct guest percentage
	correct_percentage = 0.0
	
	# max iter set to 0
	caffe_config["max_iter"] = 0
	
	# active learning loop
	iter_num = 1
	
	# allright_times -> when all item of captext_list_file fully recognized
	all_correct_times = 0
	
	# init list dataset train
	logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " init")
	createNewCaptextList()
	
	while iter_num < 10 and all_correct_times < 2:
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ==========================================")
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " iteration number {}".format(iter_num))
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ==========================================")
		
		last_snapshot = getLastSnapshot()
		
		if last_snapshot is None:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create TRAIN image set based on captext list")
			subcall(["php-cgi", "captcha.php", "dest-directory="+train_files_dir, "captext-list="+captext_list_file])
			
			# backup
			subcall(["cp", "-r", train_files_dir, train_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
		else:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " TEST pretrained snapshot " + snapshots_dir+last_snapshot['solverstate'])
			caffe_config["max_iter"] = int(last_snapshot['iteration'])
			
			# test
			# create test images
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create TEST image set based on captext list")
			subcall(["php-cgi", "captcha.php", "dest-directory="+test_files_dir, "captext-list="+captext_list_file])
			
			#bakcup
			subcall(["cp", "-r", test_files_dir, test_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
			
			correct = 0
			total_images_test = 0
			
			list_of_test_task = {}
			list_of_files = os.listdir(test_files_dir)
			total_images_test = len(list_of_files)
			
			init_task = 4 if total_images_test > 4 else total_images_test
			list_of_correct = {}
			queue_no = 1
			
			# load init 10 task
			for idx in xrange(init_task):
				each_file = list_of_files.pop(0)
				image_file_path = test_files_dir + each_file
				correct_string = os.path.splitext(each_file)[0]
				list_of_test_task[idx] = classifyImage.delay(correct_string, image_file_path, "network_captchas_with_3_convolutional_layers.prototxt", snapshots_dir+last_snapshot['caffemodel'])
				queue_no += 1
				
			#iter_ = 0
			while True:
				all_ready = True
				list_of_prediction = []
				
				for idx in xrange(init_task):
					if list_of_test_task[idx].ready():
						result = list_of_test_task[idx].get()
						
						if result[0]:
							#print("{} -> {} == {} benar".format(idx, result[1], result[2]))
							list_of_correct[ str(result[1]) ] = 0
							migrateTestFilesToRecognized(result[1], post_fix=caffe_config["max_iter"])
						else: 
							# print("{} -> {} != {} salah".format(idx, result[1], result[2]))
							pass
						
						if queue_no <= total_images_test:
							each_file = list_of_files.pop(0)
							image_file_path = test_files_dir + each_file
							correct_string = os.path.splitext(each_file)[0]
							list_of_test_task[idx] = classifyImage.delay(correct_string, image_file_path, "network_captchas_with_3_convolutional_layers.prototxt", snapshots_dir+last_snapshot['caffemodel'])
							queue_no += 1
							all_ready = False
						
					else:
						all_ready = False
				
				if all_ready:
					break
				
				#iter_ += 1
				sleep(0.7)
				
			#print("==========")
			#print(iter_)
			
			correct = len(list_of_correct)
			if correct == total_images_test:
				logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " wow all correct. 100%")
				all_correct_times += 1
				createNewCaptextList()
				correct_percentage = 100.0
			else :
				correct_percentage = correct / total_images_test * 100
				logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " booo!!. only correct: " + str(correct) + " images of total: " + str(total_images_test) )
				convertTestDirToTrainDir()
				
				#bakcup
				subcall(["cp", "-r", train_files_dir, train_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
		
		#caffe_config["max_iter"] += 50000 * (100.0 - correct_percentage)
		caffe_config["max_iter"] += len([name for name in os.listdir(train_files_dir) if os.path.isfile(os.path.join(train_files_dir, name))]) * 5
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " set new max train iter to ->" + str(caffe_config["max_iter"]) )
		
		# train
		# create train list filename and label
		subcall(["bash", "create-train-list.sh", train_files_dir, train_list_file])
		
		# reset images db
		subcall(["rm", "-rf", train_db_dir])
		
		# create new images db
		subcall(["convert_imageset", "--backend=leveldb","--gray", "--resize_height=0", "--resize_width=0", "--shuffle=true", train_files_dir, train_list_file, train_db_dir])
		cleanUpLastTrainDir()
		
		configNewNetCaffe()		
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " caffe TRAIN")
		# caffe train
		if last_snapshot is None:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " no snapshot")
			subcall(["caffe", "train", "--solver="+caffe_config_file])
		else:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " continue TRAIN snapshot " + snapshots_dir+last_snapshot['solverstate'])
			subcall(["caffe", "train", "--solver="+caffe_config_file, "--snapshot="+snapshots_dir+last_snapshot['solverstate'] ])
		
		iter_num += 1
		
		
	
if __name__ == "__main__":
	train_files_dir = "temp/train-files/"
	test_files_dir = "temp/test-files/"
	recognized_files_dir = "temp/recognized-files/"
	captext_list_file = "temp/captcha-text-list.txt"
	train_list_file = "temp/train-list.txt"
	train_db_dir = "temp/train.db"
	snapshots_dir = "temp/snapshots/"
	caffe_config_file = "captcha_solver.prototxt"
	
	caffe_config = {}	
	caffe_config["net"] = '"network_captchas_with_3_convolutional_layers_train.prototxt"'
	# The base learning rate, momentum and the weight decay of the network.
	caffe_config["base_lr"] = 0.01	
	caffe_config["momentum"] = 0.9
	caffe_config["weight_decay"] = 0.0005	
	# The learning rate policy
	caffe_config["lr_policy"] = '"inv"'
	caffe_config["gamma"] = 0.0001
	caffe_config["power"] = 0.75	
	# Display every caffe_config["display"] iterations 
	caffe_config["display"] = 500	
	# The maximum number of iterations
	caffe_config["max_iter"] = 50000	
	# snapshot intermediate results
	caffe_config["snapshot"] = 10000	
	caffe_config["snapshot_prefix"] = '"temp/snapshots/modepakbudi"'
	# solver mode: CPU or GPU
	caffe_config["solver_mode"] = 'GPU'
	
	main()
