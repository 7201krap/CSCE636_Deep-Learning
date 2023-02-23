Please follow the steps below for the setup

1. Rename the 'code' folder into 'codes'
2. In the same directory, make a folder name 'data'
	2-1. Make a folder name 'private'
		- put 'private_test_images_2022.npy'
	2-2. Make a folder name 'testing'
		- put 'test_batch' from CIFAR-10 official website
	2-3. Make a folder name 'training'
		- put from 'data_batch_1' to 'data_batch_5' from CIFAR-10 official website
3. In the same directory, make a folder name 'model_dir'
	3-1. The program will save weight here
4. In the same directory, make a folder name 'private_dir'
	4-1. Copy one of the generated models from 'model_dir' and rename it to mymodel.ckpt
5. In the same directory, make a folder name 'results'
	5-1. The program will save predictions.npy here

--------------------------------------------------------------------------------------------
Please follow the steps below to run the program

1. Training
python3 main.py --mode=train

2. Testing
python3 main.py --mode=test

3. Prediction for private dataset 
python3 main.py --mode=private