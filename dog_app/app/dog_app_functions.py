from extract_bottleneck_features import *
from os import listdir
import cv2                
import matplotlib.pyplot as plt
from keras.preprocessing import image               
from tqdm import tqdm
import numpy as np     
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential            


def path_to_tensor(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0)#.astype('float32')/255
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def initialize_model():

    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
    Xception_model.add(Dense(133, activation='softmax'))

    Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    Xception_model.load_weights('../models/weights.best.Xception.hdf5.keras')
    
    return Xception_model


def Xception_predict_breed(img_path, Xception_model):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    dog_names = ['.Affenpinscher', '.Afghan_hound', '.Airedale_terrier', '.Akita', '.Alaskan_malamute', '.American_eskimo_dog', '.American_foxhound', '.American_staffordshire_terrier', '.American_water_spaniel', '.Anatolian_shepherd_dog', '.Australian_cattle_dog', '.Australian_shepherd', '.Australian_terrier', '.Basenji', '.Basset_hound', '.Beagle', '.Bearded_collie', '.Beauceron', '.Bedlington_terrier', '.Belgian_malinois', '.Belgian_sheepdog', '.Belgian_tervuren', '.Bernese_mountain_dog', '.Bichon_frise', '.Black_and_tan_coonhound', '.Black_russian_terrier', '.Bloodhound', '.Bluetick_coonhound', '.Border_collie', '.Border_terrier', '.Borzoi', '.Boston_terrier', '.Bouvier_des_flandres', '.Boxer', '.Boykin_spaniel', '.Briard', '.Brittany', '.Brussels_griffon', '.Bull_terrier', '.Bulldog', '.Bullmastiff', '.Cairn_terrier', '.Canaan_dog', '.Cane_corso', '.Cardigan_welsh_corgi', '.Cavalier_king_charles_spaniel', '.Chesapeake_bay_retriever', '.Chihuahua', '.Chinese_crested', '.Chinese_shar-pei', '.Chow_chow', '.Clumber_spaniel', '.Cocker_spaniel', '.Collie', '.Curly-coated_retriever', '.Dachshund', '.Dalmatian', '.Dandie_dinmont_terrier', '.Doberman_pinscher', '.Dogue_de_bordeaux', '.English_cocker_spaniel', '.English_setter', '.English_springer_spaniel', '.English_toy_spaniel', '.Entlebucher_mountain_dog', '.Field_spaniel', '.Finnish_spitz', '.Flat-coated_retriever', '.French_bulldog', '.German_pinscher', '.German_shepherd_dog', '.German_shorthaired_pointer', '.German_wirehaired_pointer', '.Giant_schnauzer', '.Glen_of_imaal_terrier', '.Golden_retriever', '.Gordon_setter', '.Great_dane', '.Great_pyrenees', '.Greater_swiss_mountain_dog', '.Greyhound', '.Havanese', '.Ibizan_hound', '.Icelandic_sheepdog', '.Irish_red_and_white_setter', '.Irish_setter', '.Irish_terrier', '.Irish_water_spaniel', '.Irish_wolfhound', '.Italian_greyhound', '.Japanese_chin', '.Keeshond', '.Kerry_blue_terrier', '.Komondor', '.Kuvasz', '.Labrador_retriever', '.Lakeland_terrier', '.Leonberger', '.Lhasa_apso', '.Lowchen', '.Maltese', '.Manchester_terrier', '.Mastiff', '.Miniature_schnauzer', '.Neapolitan_mastiff', '.Newfoundland', '.Norfolk_terrier', '.Norwegian_buhund', '.Norwegian_elkhound', '.Norwegian_lundehund', '.Norwich_terrier', '.Nova_scotia_duck_tolling_retriever', '.Old_english_sheepdog', '.Otterhound', '.Papillon', '.Parson_russell_terrier', '.Pekingese', '.Pembroke_welsh_corgi', '.Petit_basset_griffon_vendeen', '.Pharaoh_hound', '.Plott', '.Pointer', '.Pomeranian', '.Poodle', '.Portuguese_water_dog', '.Saint_bernard', '.Silky_terrier', '.Smooth_fox_terrier', '.Tibetan_mastiff', '.Welsh_springer_spaniel', '.Wirehaired_pointing_griffon', '.Xoloitzcuintli', '.Yorkshire_terrier']
    return dog_names[np.argmax(predicted_vector)]


def human_dog_classifier(img_path, model):
    '''
    The function checks, whether the provided image is a human, a dog or neither of the two and returns the most resembling dog breed for dogs and humans.

    INPUT:
    img_path: path to a photo to be classified
    
    OUTPUT:
    None
    '''
    print ('-----------------------------------------------------------------------------')
    
    # Display Image
    im = cv2.imread(img_path)
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    #plt.show()
    print(img_path)

    human = False
    dog = True

    # Check whether the image shows a human
    if human == True:
        print('The image shows a human.')
    
    #Check whether the image shows a dog
    if dog == True:
        print('The image shows a dog.')
    
    if (human == True) | (dog == True):

        breed_string=Xception_predict_breed(img_path, model)[1:]
        breed = breed_string.replace("_"," ")
        breed_path ="static/" + breed_string + ".jpg"
        result_string = f' The image resembles most to a {breed}.'

        #dog_breed_image(breed)    
        return breed, result_string, breed_path
    
    if (human == False) & (dog == False):
        print('ERROR: I cannot classify the picture as human or dog') 
        return None, None, None