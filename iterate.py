from subprocess import call
from random import randint

def modePakBudiCaptext():
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	captext_list = []
	
	# create captcha text modepakbudi sebanyak 10 kali
	for iter_captext in xrange(10):
		
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
	print("create new captext_list")
	captext_list = modePakBudiCaptext()
	
	# append captext_list to captext_list_file
	with open(captext_list_file, "a") as myfile:
		for each_text in captext_list:
			myfile.write(each_text)

def resetCaptexList():
	# reset list text captcha
	call(["rm", "-rf", captext_list_file])
	call(["touch", captext_list_file])			

def cleanUpDir(dir_path):
	call(["rm", "-rf", dir_path])
	call(["mkdir", dir_path])
	call(["touch", dir_path+"supayakeadd"])

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
	
	params["snapshot_prefix"] = 5000
	
	# solver mode: CPU or GPU
	params["solver_mode"] = "GPU"
	
	return params



	

def main():
	print("clean up first")
	resetCaptexList()	
	cleanUpLastTrainDir()
	cleanUpLastTestDir()
	
	# init list dataset train
	print("init captext list")
	createNewCaptextList()
	
	# active learning loop
	iter_num = 1
	
	# allright_times -> when all item of captext_list_file fully recognized
	allright_times = 0
	
	while iter_num < 3 and allright_times < 2:
		print("iteration number {}".format(iter_num))
		print("==========================================")
		print("create TRAIN image set based on captext list")
		call(["php-cgi", "captcha.php", "dest-directory="+train_files_dir, "captext-list="+captext_list_file])
		
		# reset images db
		call(["rm", "-rf", train_db_dir])
		# create new images db
		call(["convert_imageset", "--gray", "--resize_height=0", "--resize_width=0", "--shuffle=true", train_files_dir, captext_list_file, train_db_dir])
		
		cleanUpLastTrainDir()
		
		# caffe train
		call(["caffe", "train", "--solver=captcha_solver.prototxt"])
		
		print("create TEST image set based on captext list")
		call(["php-cgi", "captcha.php", "dest-directory="+test_files_dir, "captext-list="+captext_list_file])
		
		# test
		# nanti tambahin di bawah sini
		allright = True
		
		for each_text in xrange(0,260):
			acak = radint(0,9)
			if acak > 2:
				# anggap bagian ini berarti dy masih salah baca
				allright = False
				pass
			else:
				# anggap bagian ini berarti dy udah bisa baca
				migrateTestFilesToRecognized(each_text)
		
		if allright:			
			print("wow all right. ")
			allright_times += 1
			createNewCaptextList()
		else :
			convertTestDirToTrainDir()
		
		iter_num += 1
		
		
	
if __name__ == "__main__":
	train_files_dir = "temp/train-files/"
	test_files_dir = "temp/test-files/"
	recognized_files_dir = "temp/recognized-files/"
	captext_list_file = "temp/captcha-text-list.txt"
	train_db_dir = "temp/train.db"
	main()
