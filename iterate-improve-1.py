import pdb
import argparse
import datetime
import sys
from subprocess import call, check_output
from random import randint
from copy import copy, deepcopy
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

def additionalText(file_input, total_num_text):
	subcall(["python", "letter_analyze.py", file_input, "--destdir", "temp/"])
	file_name = file_input.split('/')[-1].split('.')[0]
	file_output = "temp/" + file_name + '_suppose_tobe_letter.csv'
	incorrect_occurance = 0
	data_list = []
	
	with open(file_output) as fp:
		for i, line in enumerate(fp):
			if i > 0:
				tmp_split = line.split(',')
				data = []
				data.append(tmp_split[0])
				data.append(tmp_split[1])
				data.append(int(tmp_split[2]))
				data_list.append(copy(data))
				incorrect_occurance += int(tmp_split[2])
	# import pdb; pdb.set_trace()
	captext_list = []
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	captext_len = 6
	captext_total = 0
	
	for data in data_list:
		num_text = int ( ( float(data[2]) / incorrect_occurance )  * total_num_text )		
		
		for i in xrange(num_text):
			captext = ""
			a_rand = 0; b_rand = 0		
			
			# pick index where each_letter would be in		
			a_rand = randint(0, captext_len - 1)
			while a_rand == b_rand:
				b_rand = randint(0, captext_len - 1)
			
			for letter_id in xrange(captext_len):
				random_letter = alphabet[randint(0, len(alphabet) - 1 )]
				
				if a_rand == letter_id:
					captext += data[0]
				elif b_rand == letter_id:
					captext += data[1]
				else: captext += random_letter
			captext_list.append(captext)
			captext_total += 1
	
	subcall(["echo", "{} of registered additional captext".format(captext_total) ])
	subcall(["rm", "-rf", additional_captext_list_file])
	subcall(["touch", additional_captext_list_file])
	
	# append captext_list to captext_list_file
	for each_text in captext_list:
		call("echo {} >> {}".format(each_text, additional_captext_list_file), shell="True")

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
	# call(list_)
	output = check_output(list_)
		
	if output is not None and output != "":
		# logger.info(output)
		# raise Exception('ada yang error')
		pass
	return output

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

def resetTrainList():
	# reset list text captcha
	subcall(["rm", "-rf", train_list_file])
	subcall(["touch", train_list_file])

def cleanUpDir(dir_path, remake=True):
	subcall(["rm", "-rf", dir_path])
	
	if remake:
		subcall(["mkdir", dir_path])
	#subcall(["touch", dir_path+"supayakeadd"])

def cleanUpRecognizedDir():
	cleanUpDir(recognized_files_dir)
	subcall(["touch", recognized_files_dir+"supayakeadd"])

def migrateTestFilesToRecognized(full_path, post_fix=None):
	file_name = full_path.split('/')[-1]

	post_fix = "__"+str(post_fix)
	
	call(["mv", full_path, recognized_files_dir+file_name+post_fix])

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

def normalizeSnapshotDir(last_max_iteration=1):
	if last_max_iteration < 2:
		cleanUpDir(snapshots_dir)
		return
	
	while True:
		last_snapshot = getLastSnapshot()
		if int(last_snapshot['iteration']) > int(last_max_iteration):
			subcall(['rm', '-rf', snapshots_dir+last_snapshot['solverstate'] ])
			subcall(['rm', '-rf', snapshots_dir+last_snapshot['caffemodel'] ])
		else : return

def normalizeTrainDir(last_max_iteration=1):
	output = check_output('ls -1td ' + train_files_dir[0:-1]+'*', shell=True)
	output = output.split('\n')
	
	for each_dir in output:
		tmp_split = each_dir.split('_')
		if each_dir.find('__bak_') > -1 and (int(tmp_split[-1]) >= last_max_iteration or last_max_iteration < 2):
			subcall(['rm', '-rf',  each_dir])
			
	cleanUpDir(train_files_dir)

def normalizeTestDir(last_max_iteration=1):
	output = check_output('ls -1td ' + test_files_dir[0:-1]+'*', shell=True)
	output = output.split('\n')	
	
	for each_dir in output:
		tmp_split = each_dir.split('_')
		if each_dir.find('__bak_') > -1 and int(tmp_split[-1]) >= last_max_iteration:
			subcall(['rm', '-rf',  each_dir])
			
	cleanUpDir(test_files_dir)

def saveGlobalConfig():
	subcall(["rm", "-rf", global_config_file])
	subcall(["touch", global_config_file])
	
	for each_ in global_config:
		call("echo {}:{} >> {}".format(each_, global_config[each_], global_config_file), shell="True")

def loadGlobalConfig():
	with open(global_config_file) as f:
		content = f.readlines()
		
	for each_line in content:
		each_line = each_line.strip()
		tmp_split = each_line.split(':')
		global_config[tmp_split[0] ] = float(tmp_split[1]) if tmp_split[1].find('.') > -1 else int (tmp_split[1])
	

#def classifyImage(input_image, network, caffe_model):
#	classifier = caffe.Classifier(network, caffe_model, mean=None)
#	input_images = [input_image]
#	
#	prediction = classifier.predict(input_images, oversample=False)
#	return prediction

def main():
	# correct guest percentage
	# global_config['last_correct_percentage'] = 0.0
	
	# max iter set to 0
	caffe_config["max_iter"] = 0
	
	# active learning loop
	# global_config['iteration_number'] = 1
	
	# allright_times -> when all item of captext_list_file fully recognized
	# global_config['all_correct_times'] = 0
	
	if args.reset:
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " Init")
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " clean up first")
		cleanUpRecognizedDir()
		createNewCaptextList()
	
	else:
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " Continue")
		loadGlobalConfig()
		caffe_config["max_iter"] = global_config['last_caffe_iteration'] 
	
	resetTrainList()
	normalizeSnapshotDir(last_max_iteration=global_config['last_caffe_iteration'])
	normalizeTestDir(last_max_iteration=global_config['last_caffe_iteration'])
	normalizeTrainDir(last_max_iteration=global_config['last_caffe_iteration'])
	
	while global_config['iteration_number'] < 30 and global_config['all_correct_times'] < 3:
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ==========================================")
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " iteration number {}".format(global_config['iteration_number']))
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ==========================================")
		
		last_snapshot = getLastSnapshot()
		
		if last_snapshot is None:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " No snapshot. create TRAIN image set based on captext list")
			subcall(["php-cgi", "captcha-hash.php", "dest-directory="+train_files_dir, "captext-list="+captext_list_file])
			
			# backup
			subcall(["cp", "-r", train_files_dir, train_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
		else:
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " TEST pretrained snapshot " + snapshots_dir+last_snapshot['solverstate'])
			caffe_config["max_iter"] = int(last_snapshot['iteration'])
			
			# test
			# create test images
			logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create TEST image set based on captext list")
			subcall(["php-cgi", "captcha-hash.php", "dest-directory="+test_files_dir, "captext-list="+captext_list_file])
			
			#bakcup
			subcall(["cp", "-r", test_files_dir, test_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
			
			correct = 0
			total_images_test = 0
			
			list_of_test_task = {}
			list_of_files = os.listdir(test_files_dir)
			total_images_test = len(list_of_files)
			
			init_task = 4 if total_images_test > 4 else total_images_test
			list_of_correct = {}
			queue_no = 0
			redun_correct = 0
			
			subcall(["rm", "-rf", result_csv])
			subcall(["touch", result_csv])
			
			# load init task
			for idx in xrange(init_task):
				each_file = list_of_files.pop(0)
				image_file_path = test_files_dir + each_file
				correct_string = os.path.splitext(each_file)[0]
				list_of_test_task[idx] = classifyImage.delay(correct_string, image_file_path, "network_captchas_with_3_convolutional_layers.prototxt", snapshots_dir+last_snapshot['caffemodel'])
				queue_no += 1			
			
			while True:
				all_ready = True
				list_of_prediction = []
				
				for idx in xrange(init_task):
					if idx not in list_of_test_task:
						if len(list_of_files) > 0:
							each_file = list_of_files.pop(0)
							image_file_path = test_files_dir + each_file
							correct_string = os.path.splitext(each_file)[0]
							list_of_test_task[idx] = classifyImage.delay(correct_string, image_file_path, "network_captchas_with_3_convolutional_layers.prototxt", snapshots_dir+last_snapshot['caffemodel'])
							queue_no += 1
							all_ready = False
						continue
					
					# when task's result is ready
					if list_of_test_task[idx].ready():
						result = deepcopy( list_of_test_task.pop(idx).get() )
						
						if result[0]:
							#print("{} -> {} == {} benar".format(idx, result[1], result[2]))
							migrateTestFilesToRecognized(result[3], post_fix=caffe_config["max_iter"])
							list_of_correct[ result[3].split('/')[-1] ] = 0
							redun_correct += 1
						else:
							# print("{} -> {} != {} salah".format(idx, result[1], result[2]))
							pass
						
						call("echo {},{},{},{},{} >> {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(result[0]), result[1], result[2], caffe_config["max_iter"], result_csv), shell="True")
						
						# assign new queue task
						if len(list_of_files) > 0:
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
			tmp_split = result_csv.split(".")
			result_bak_csv = tmp_split[0] + "_" + str(caffe_config["max_iter"]) + "." + tmp_split[1]
			subcall(["mv", result_csv, result_bak_csv])
			correct = len(list_of_correct)
			
			if correct == total_images_test:
				logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " wow all correct. 100%")
				global_config['all_correct_times'] += 1
				# backup last captext list
				subcall(["cp", "-r", captext_list_file, captext_list_file + "__bak_" + str(caffe_config["max_iter"])])
				createNewCaptextList()
				global_config['last_correct_percentage'] = 100.0
			else :
				global_config['last_correct_percentage'] = float(correct) / float(total_images_test) * 100.0
				logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " booo!!. only correct: " + str(correct) + " images of total: " + str(total_images_test) )
				convertTestDirToTrainDir()
				
				additionalText(result_bak_csv, redun_correct)
				logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " create additional captcha")
				subcall(["php-cgi", "captcha-hash.php", "dest-directory="+train_files_dir, "captext-list="+additional_captext_list_file])
				
				# bakcup train files
				subcall(["cp", "-r", train_files_dir, train_files_dir[0:-1] + "__bak_" + str(caffe_config["max_iter"])])
			
			logger.info("{} queue total -> {}, total image set -> {}, redun_correct -> {}, correct_percentages -> {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), queue_no, total_images_test, redun_correct, global_config['last_correct_percentage']))
		
		#caffe_config["max_iter"] += 50000 * (100.0 - global_config['last_correct_percentage'])
		caffe_config["max_iter"] += len([name for name in os.listdir(train_files_dir) if os.path.isfile(os.path.join(train_files_dir, name))]) * 6
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " set new max train iter to ->" + str(caffe_config["max_iter"]) )
		
		# train
		# create train list filename and label
		resetTrainList()
		
		subcall(["bash", "create-train-list-ext.sh", train_files_dir, train_list_file])
		
		if global_config['last_correct_percentage'] < 60.0:
			subcall(["bash", "create-train-list-ext.sh", train_files_dir, train_list_file])		
			pass
		
		if global_config['last_correct_percentage'] < 50.0: #or global_config['last_correct_percentage'] > 75.0
			subcall(["bash", "create-train-list-ext.sh", train_files_dir, train_list_file])
			pass
		
		# reset images db
		subcall(["rm", "-rf", train_db_dir])
		
		# create new images db
		subcall(["convert_imageset", "--backend=leveldb","--gray", "--resize_height=0", "--resize_width=0", "--shuffle=true", train_files_dir, train_list_file, train_db_dir])
		# cleanUpLastTrainDir()
		cleanUpDir(train_files_dir)
		
		configNewNetCaffe()		
		logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " caffe TRAIN")
		
		# caffe train		
		temp_attempt = 0
		
		while True:			
			if temp_attempt > 2:
				break
			else: sleep(60 * 5); temp_attempt += 1
			
			try:
				if last_snapshot is None:
					logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " no snapshot")
					subcall(["caffe", "train", "--solver="+caffe_config_file])
				else:
					logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " continue TRAIN snapshot " + snapshots_dir+last_snapshot['solverstate'])
					subcall(["caffe", "train", "--solver="+caffe_config_file, "--snapshot="+snapshots_dir+last_snapshot['solverstate'] ])
				break
			except Exception as e:
				logger.error(str(e))
		
		global_config['last_caffe_iteration'] = caffe_config["max_iter"]
		global_config['iteration_number'] += 1
		saveGlobalConfig()

"""if __name__ == "__main__":
	result_bak_csv = 'temp/result_60060.csv'
	additional_captext_list_file = 'temp/additional-captcha-text-list.txt'
	additionalText(result_bak_csv, 100)
"""	
if __name__ == "__main__":
	train_files_dir = "temp/train-files/"
	test_files_dir = "temp/test-files/"
	recognized_files_dir = "temp/recognized-files/"
	captext_list_file = "temp/captcha-text-list.txt"
	additional_captext_list_file = "temp/additional-captcha-text-list.txt"
	train_list_file = "temp/train-list.txt"
	train_db_dir = "temp/train.db"
	snapshots_dir = "temp/snapshots/"
	caffe_config_file = "captcha_solver.prototxt"
	result_csv = "temp/result.csv"
	
	global_config_file = "config.txt"
	
	global_config = {}
	global_config['iteration_number'] = 1
	global_config['last_correct_percentage'] = 0.0
	global_config['all_correct_times'] = 0
	global_config['last_caffe_iteration'] = 0
	
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
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--reset", help="mulai dari awal", action="store_true")
	args = parser.parse_args()
	
	main()

