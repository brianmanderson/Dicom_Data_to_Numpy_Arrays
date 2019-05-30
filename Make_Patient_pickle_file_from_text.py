from Utils import os, save_obj


def main_run(images_description = 'Bastien_Cervix_Uterus_Data_No_Applicator',
             base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Bastien'):


    path = os.path.join(base_path,'Numpy_' + images_description)
    Series_Descriptions = []
    for _, Series_Descriptions, _ in os.walk(path):
        break

    patient_spacing_info = {}
    for Series_Description in Series_Descriptions:
        patient_spacing_info[Series_Description] = {}
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
    save_obj(os.path.join(base_path,'patient_info_' + images_description + '.pkl'), patient_spacing_info)

if __name__ == '__main__':
    xxx = 1