# cocktail-party-deep-learning
Cocktail party problem solution using deep learning

Guide to installations and running the project:

For Linux (Ubuntu):

1. Install the python-tk package by typing the following command:
	sudo apt-get install python-tk

2. Then install the following:
	sudo apt-get install python-dev python-pip

3. Then install the folllwing python libraries:
	sudo pip install tensorflow
	sudo pip install librosa
	sudo pip install matplotlib
	sudo pip install numpy
	
To run the main program:
	Run 'guitar_separate_test_CNN.py' in python using the command:
		python guitar_separate_test_CNN.py audio_file1.wav audio_file2.wav
		(note: Both files should be more than 6 seconds long)
	
	If you have only one file which contains mixed music already:
		python guitar_seaparate_test_CNN.py audio_mix.wav no


	
