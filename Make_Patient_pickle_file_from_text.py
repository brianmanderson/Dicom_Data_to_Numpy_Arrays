import os, pickle


def save_obj(obj, path): # Save almost anything.. dictionary, list, etc.
    if path[-4:] != '.pkl':
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
    return None
def load_obj(path):
    if path[-4:] != '.pkl':
        path += '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def main_run(images_description = 'Bastien_Cervix_Uterus_Data_No_Applicator',
             base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Bastien'):


    path = os.path.join(base_path,'Numpy_' + images_description)
    Series_Descriptions = []
    for _, Series_Descriptions, _ in os.walk(path):
        break

    patient_spacing_info = {}
    for Series_Description in Series_Descriptions:
        patient_spacing_info[Series_Description] = {}
        Series_Descriptions = []
        for _, _, files in os.walk(os.path.join(path,Series_Description)):
            break
        file_list = [i for i in files if i.find('.txt') != -1 and i.find('Iteration') != -1]
        for file in file_list:
            iteration = (file.split('Iteration_')[-1]).split('.txt')[0]
            fid = open(os.path.join(path,Series_Description,file))
            data = fid.readline()
            fid.close()
            data = data.strip('\n')
            data = data.split(',')
            patient_spacing_info[Series_Description][iteration] = data[0] + ',' + data[1] + ',' + data[2]
    save_obj(patient_spacing_info,os.path.join(base_path,'patient_info_' + images_description + '.pkl'))

if __name__ == '__main__':
    xxx = 1