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

def cleanUpLastTrainDir():
	# reset directory train image captcha
	call(["rm", "-rf", train_files_dir])
	call(["mkdir", train_files_dir])
	call(["touch", train_files_dir+"supayakeadd"])	
	

def main():	
	# reset list text captcha
	call(["rm", "-rf", captext_list_file])
	call(["touch", captext_list_file])
	
	cleanUpLastTrainDir()
	
	captext_list = modePakBudiCaptext()
	
	# append captext_list ke dalam file
	with open(captext_list_file, "a") as myfile:
		for each_text in captext_list:
			myfile.write(each_text)
	
	# create train image captcha based on captext_list_file
	call(["php-cgi", "captcha.php", "dest-directory="+train_files_dir, "captext-list="+captext_list_file])
	
	
	# reset db directory
	call(["rm", "-rf", train_db_dir])
	
	# convert images into db
	call(["convert_imageset", "--gray", "--resize_height=0", "--resize_width=0", "--shuffle=true", train_files_dir, captext_list_file, train_db_dir])
	
	# caffe train
	call(["caffe", "train", "--solver=captcha_solver.prototxt"])
	
if __name__ == "__main__":
	train_files_dir = "temp/train-files/"
	captext_list_file = "temp/captcha-text-list.txt"
	train_db_dir = "temp/train.db"
	main()
