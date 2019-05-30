import os
import numpy as np
import pickle

def load_obj(path):
    if path[-4:] != '.pkl':
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def separate(desc='Bastien_Cervix_Uterus_Data_No_Applicator',path_base=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Bastien'):
    path = os.path.join(path_base,'Numpy_' + desc)
    pickle_file = [file for file in os.listdir(path_base) if file.find('.pkl') != -1 and file.find(desc) != -1][0]
    patient_info = load_obj(os.path.join(path_base,pickle_file))
    patient_dict = patient_info[list(patient_info.keys())[0]]
    keys = list(patient_dict.keys())
    if 'path' in keys:
        del keys[keys.index('path')]
    all_images = os.path.join(path,'CT')
    train_path = os.path.join(path,'Train')
    test_path = os.path.join(path, 'Test')
    validation_path = os.path.join(path,'Validation')
    for out_path in [train_path,test_path,validation_path]:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    image_dict = dict()
    i = 0
    patient_file_associations = {}
    file_list = [i for i in os.listdir(all_images) if i.find('.npy') != -1]
    for file in file_list:
        if file.find('Overall_Data') == 0:
            continue
        mask = file
        image = file.replace('Overall_mask','Overall_Data')
        image_end = image.split('_')[-1][1:]
        image = image.split('_')
        image = [i + '_' for i in image]
        image[-1] = image_end
        image = ''.join(image)
        image_dict[i] = [image,mask]
        iteration = image.split('_')[-1].split('.npy')[0]
        print(file)
        patient_MRN = patient_dict[iteration].split('\\')[0].split(',')[0]
        if patient_MRN not in patient_file_associations.keys():
            patient_file_associations[patient_MRN] = []
        patient_file_associations[patient_MRN].append(image_dict[i])
        i += 1
    patient_image_keys = list(patient_file_associations.keys())
    perm = np.arange(len(patient_image_keys))
    np.random.shuffle(perm)
    patient_image_keys = list(np.asarray(patient_image_keys)[perm])
    split_train = int(len(patient_image_keys)/6)
    for xxx in range(split_train):
        for image,mask in patient_file_associations[patient_image_keys[xxx]]:
            os.rename(os.path.join(all_images,image),os.path.join(test_path,image))
            os.rename(os.path.join(all_images, mask), os.path.join(test_path, mask))
    for xxx in range(split_train,int(split_train*2)):
        for image,mask in patient_file_associations[patient_image_keys[xxx]]:
            os.rename(os.path.join(all_images,image),os.path.join(validation_path,image))
            os.rename(os.path.join(all_images, mask), os.path.join(validation_path, mask))
    for xxx in range(int(split_train*2),len(perm)):
        for image,mask in patient_file_associations[patient_image_keys[xxx]]:
            os.rename(os.path.join(all_images,image),os.path.join(train_path,image))
            os.rename(os.path.join(all_images, mask), os.path.join(train_path, mask))

if __name__ == '__main__':
    xxx = 1
    separate(desc='',path_base=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Pancreas\Pancreas\cancer_imaging_archive_Data\Numpy')