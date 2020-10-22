# This will split our dataset into training, validation, and testing sets in the ratio mentioned above- 80% for training (of that, 10% for validation) and 20% for testing. With the ImageDataGenerator from Keras, we will extract batches of images to avoid making space for the entire dataset in memory at once.

from cancernet import config
from imutils import paths
import random, shutil, os

originalPaths = list(paths.list_images(config.INPUT_DATASET))
random.seed(7)
random.shuffle(originalPaths)

index=int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths = originalPaths[:index]
testPaths = originalPaths[index:]

index = int(len(trainPaths)*config.VAL_SPLIT)
valPaths = trainPaths[:index]
trainPaths = trainPaths[index:]

datasets = [("training",trainPaths,config.TRAIN_PATH),
            ("validation",valPaths,config.VAL_PATH),
            ("testing",testPaths,config.TEST_PATH)
            ]

for(setType,originalPaths,basePath) in datasets:
    print(f'Building {setType} set')

    if not os.path.exists(basePath):
        print(f'Building directory {basePath}')
        os.makedirs(basePath)

    for path in originalPaths:
        file=path.split(os.path.sep)[-1]
        label=file[-5:-4]
        print("label = "+str(label))

        labelPath = os.path.sep.join([basePath,label])
        if not os.path.exists(labelPath):
            print(f'Building directory {labelPath}')
            os.makedirs(labelPath)

        print("labelPath = "+str(labelPath))
        newPath=os.path.sep.join([labelPath, file])
        print("Old path = " +str(path)+ " new path ="+ str(newPath))
        shutil.copy2(path,newPath)
