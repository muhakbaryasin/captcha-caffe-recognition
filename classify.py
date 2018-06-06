from task_async import app
import caffe
from celery.contrib import rdb

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


@app.task
def classifyImage(correct_string, image_file_path, network, caffe_model):
	correct_string = str(correct_string)
	image_file_path = str(image_file_path)
	network = str(network)
	caffe_model = str(caffe_model)
	input_image = caffe.io.load_image(image_file_path, color=False)
	classifier = caffe.Classifier(network, caffe_model, mean=None)
	input_images = [input_image]	
	prediction = classifier.predict(input_images, oversample=False)
	
	predictedString = ""
	numberOfDigits = 6
	classesPerDigit = 63
	
	for x in xrange(0, numberOfDigits):
		predictedChar = prediction[0][63*x:63*(x+1)]
		predictedChar = predictedChar * sum(predictedChar) ** -1
		predictedClass = predictedChar.argmax()
		predictedCharacter = convertClassToCharacter(predictedClass)
		predictedString+=predictedCharacter
	
	if predictedString == correct_string:
		return (True, predictedString, correct_string)
	else: return (False, predictedString, correct_string)
