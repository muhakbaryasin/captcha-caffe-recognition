import argparse
from subprocess import call
from collections import OrderedDict

def main():
	call(['touch', wrong_guest_letter])
	call(['touch', suppose_tobe_letter])
	
	call("echo {},{},{} >> {}".format('guest', 'suppose_tobe', 'number', suppose_tobe_letter), shell="True")
	
	suppose_tobe_dict = {}
	wrong_guest_dict = {}
	
	with open(file_path) as fp:
		for line in fp:
			tmp_split = line.split(',')
			
			if tmp_split[1] == 'True':
				continue
			
			idx = 0
			
			for each_letter in tmp_split[2]:
				correct = tmp_split[3][idx]
				
				if each_letter != correct:
					key = each_letter+'->'+correct
					
					if key in suppose_tobe_dict:
						suppose_tobe_dict[key] += 1
					else: suppose_tobe_dict[key] = 1
					
					if correct in wrong_guest_dict:
						wrong_guest_dict[correct] += 1
					else:
						wrong_guest_dict[correct] = 1
						
				idx += 1
	
	suppose_tobe_dict = OrderedDict(sorted(suppose_tobe_dict.items(), key=lambda t: t[1], reverse=True))
	wrong_guest_dict = OrderedDict(sorted(wrong_guest_dict.items(), key=lambda t: t[1], reverse=True))
	
	for each_ in suppose_tobe_dict:
		tmp_split = each_.split('->')		
		call("echo {},{},{} >> {}".format(tmp_split[0], tmp_split[1], suppose_tobe_dict[each_], suppose_tobe_letter), shell="True")
	
	for each_ in wrong_guest_dict:
		call("echo {},{} >> {}".format(each_, wrong_guest_dict[each_], wrong_guest_letter), shell="True")
				

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", help="inputan file dari result_xx program captcha caffe akbar", type=str)
	parser.add_argument("--destdir", help="directory tujuan", type=str)
	args = parser.parse_args()

	path = args.destdir

	if path[-1] != '/':
		path += '/'
	
	file_path = args.filepath
	file_name = file_path.split('/')[-1].split('.')[0]

	suppose_tobe_letter = path + file_name + '_suppose_tobe_letter.csv'
	wrong_guest_letter = path + file_name + '_wrong_guest_letter.csv'
	main()
