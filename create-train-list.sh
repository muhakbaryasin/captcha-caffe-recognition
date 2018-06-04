TRAIN_DATA_ROOT=$1
TRAIN=$2

#TRAIN_DATA_ROOT=/home/go/Project/captcha-caffe-recognition/temp/train-files/
#TRAIN=/home/go/Project/captcha-caffe-recognition/temp/train.txt

# This function maps a character to a number.
# 0 -> 0, 1->1, ... 9->9, A->10, B->11, ... Z->35, 
# a->37, b->38, ... z->62
# there is a small mistkate! The class 36 is never asigned. But it doesn't matter ;)
convertCharacterToOutput(){
	ascii_value=$(printf '%d' "'$1")
	# a small letter
	if [ "$ascii_value" -gt 90 ]; then
		value=$((ascii_value-60))
	else
		# a big letter
		if [ "$ascii_value" -gt 57 ]; then
			value=$((ascii_value-55))
		# a digit
		else
			value=$((ascii_value-48))
		fi 
	fi
	return $value
}

for name in $TRAIN_DATA_ROOT*; do
	filename=$(basename "$name")
	filename="${filename%.*}"
	captcha_length=${#filename}
	for (( i=0; i<$captcha_length; i++ )); do
		character=${filename:$i:1}
		convertCharacterToOutput $character
		asciivalue=$?
		asciivalue=$((asciivalue + 63*i))
		echo $filename" "$asciivalue >> $TRAIN
		
	done
done
