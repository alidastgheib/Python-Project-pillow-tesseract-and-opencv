#%%
# Author: Ali Dastgheib
# This code tries to find the most suitable parameters for 
# cv::CascadeClassifier::detectMultiScale(   scaleFactor , minNeighbors   )

#%%
import zipfile
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os

#%%
def create_images_dict_from_zip (zip_dir = None):
    '''
    This function creates a dictionary of PIL image objects that are
    images inside a ZIP file. The address of this ZIP file is the input
    of this function. Keys of the dictionary are the image files' names.
    '''
    zf = zipfile.ZipFile(zip_dir, 'r')
    print(zf.namelist())
    images_dict = {}
    for fileName in zf.namelist():
        try:
            tempZip = zf.open(fileName)
            tempIMG = Image.open(tempZip)
            images_dict[fileName] = tempIMG
        except KeyError:
            print ('ERROR: Did not find %s in zip file' % fileName) 
        else:
            print (fileName + ' is read successfully.')
    return images_dict

#%%
zip_dir = 'small_img.zip'
images_dict = create_images_dict_from_zip(zip_dir)

#%%    
def parameter_search_for_detectMultiScale(images_dict, 
                                          face_xml_dir='haarcascade_frontalface_default.xml',
                                          face_scaleFactor=1.1, 
                                          face_min_neighbor=3):
    '''
    This function performs the "detectMultiScale" function and writes the resulting 
    cropped patches to a specified folder. The folder's name is chosen based on the 
    values of 'face_scaleFactor' and 'face_min_neighbor'.
    '''
    face_cascade = cv.CascadeClassifier(face_xml_dir)
    cropped_patches_dir = 'cropped-patches\\' + 'SF=' + str(face_scaleFactor) + \
    '--minN=' + str(face_min_neighbor) + '\\'
    
    if not os.path.exists(cropped_patches_dir): os.makedirs(cropped_patches_dir)

    for fileName, image_elem in images_dict.items():
        temp_faces_pos_list = face_cascade.detectMultiScale(np.array(image_elem), 
                                                            scaleFactor=face_scaleFactor, 
                                                            minNeighbors=face_min_neighbor)
        
        # print('temp_faces_pos_list:\n', temp_faces_pos_list)
        counter = 0
        for face_vec in temp_faces_pos_list:
            try:
                face_image_crop = np.array(image_elem)[face_vec[1]:face_vec[1] + face_vec[3], 
                                              face_vec[0]:face_vec[0] + face_vec[2], 
                                              :]
            except:
                face_image_crop = np.array(image_elem)[face_vec[1]:face_vec[1] + face_vec[3], 
                                          face_vec[0]:face_vec[0] + face_vec[2]]
                
            Image.fromarray(face_image_crop).save(cropped_patches_dir + \
                           fileName.split('.')[0] + '--' + \
                           str(counter) + '.jpg')
            counter += 1

        # print(fileName, face_scaleFactor, face_min_neighbor)
    # print(face_scaleFactor, face_min_neighbor, 'IS DONE.')
    return 0
                
#%%
# SF_list = [1.01, 1.05, 1.1, 1.2, 1.3, 1.4] # for round one
SF_list = [1.1, 1.13, 1.16, 1.19] # for round two
SF_list.sort(reverse=True) # for performing shorter-runs first
# min_Neigh_list = [3, 4, 5, 6, 7] # for round one
min_Neigh_list = [8, 9, 10] # for round two

# Crazy problems occur when SF_elem=1.01; python doesn't raise any error, however,
# in this case, python stops at the end of the function and doesn't run the rest
# of the code. Python does't come back to the for loop anymore. 

for SF_elem in SF_list:
    for min_Neigh_elem in min_Neigh_list:
        time_now = time.time()
        parameter_search_for_detectMultiScale(images_dict=images_dict, 
                                              face_scaleFactor=SF_elem, 
                                              face_min_neighbor=min_Neigh_elem)
        # print('we are inside for')
        elapsed_time = time.time() - time_now
        print('SF_elem =', SF_elem, 'min_Neigh_elem =', min_Neigh_elem)
        print('elapsed_time =', elapsed_time)

#%%
print('After analysing the resulting cropped images manually, we see that:')
print('scaleFactor = 1.16 , minNeighbors = 10')
print('results into one of the best responses.')
     
#%%
# # The script (pre-code) of the function "parameter_search_for_detectMultiScale"
#face_xml_dir = 'haarcascade_frontalface_default.xml'
#face_cascade = cv.CascadeClassifier(face_xml_dir)
#
#face_scaleFactor = 1.1
#face_min_neighbor = 5
#cropped_patches_dir = 'cropped-patches--SF=' + str(face_scaleFactor) + \
#'--minN=' + str(face_min_neighbor) + '\\'
#
#if not os.path.exists(cropped_patches_dir): os.makedirs(cropped_patches_dir)
#for fileName, image_elem in images_dict.items():
#    temp_faces_pos_list = face_cascade.detectMultiScale(np.array(image_elem), 
#                                                        scaleFactor=face_scaleFactor, 
#                                                        minNeighbors=face_min_neighbor)
#    
#    counter = 0
#    for face_vec in temp_faces_pos_list:
#        try:
#            face_image_crop = np.array(image_elem)[face_vec[1]:face_vec[1] + face_vec[3], 
#                                          face_vec[0]:face_vec[0] + face_vec[2], 
#                                          :]
#        except:
#            face_image_crop = np.array(image_elem)[face_vec[1]:face_vec[1] + face_vec[3], 
#                                          face_vec[0]:face_vec[0] + face_vec[2]]
#    
#        Image.fromarray(face_image_crop).save(cropped_patches_dir + fileName + '--' + 
#                        str(counter) + '.jpg')
#        counter += 1
