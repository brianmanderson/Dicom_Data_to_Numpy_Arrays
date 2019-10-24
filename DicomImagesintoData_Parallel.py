import pydicom, os, sys
sys.path.append('.')
from pydicom.tag import Tag
import numpy as np
from skimage import draw
from threading import Thread
from multiprocessing import cpu_count
from queue import *
from Utils import load_obj, plot_scroll_Image
from Make_Patient_pickle_file_from_text import main_run
from Separate_Numpy_Images_Into_Test_Train_Validation import separate
from Get_Path_Info import make_location_pickle
import SimpleITK as sitk

def correct_association_file(associations):
    '''
    :param associations: dictionary of associations
    :return: dictionary with keys and results lower-cased
    '''
    new_associations = {}
    for key in associations:
        new_associations[key.lower()] = associations[key].lower()
    return new_associations

def worker_def(A):
    q, associations, Contour_Names, ignore_lacking = A
    check_RS = Check_RS_Structure(associations, Contour_Names, ignore_lacking)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                check_RS.prep_data(item)
            except:
                print('failed?')
            q.task_done()

def worker_def_write(A):
    q, Contour_Names, Contour_Key, path, data_path, images_description, associations, argmax = A
    Dicom_data_class = DicomImagestoData(Contour_Names=Contour_Names, Contour_Key=Contour_Key,
                      path=path,
                      data_path=data_path,images_description=images_description,
                      associations=associations, argmax=argmax)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                Dicom_data_class.Make_Contour_From_directory(item)
            except:
                print(item)
                print('failed?')
            q.task_done()

class Check_RS_Structure(object):
    def __init__(self, associations={}, Contour_Names=[], ignore_lacking=False):
        self.associations = associations
        self.ignore_lacking= ignore_lacking
        self.Contour_Names = Contour_Names
        self.mask_exist = False

    def prep_data(self,PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = []
        self.Dicom_Data = []

        fileList = []
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        if len(fileList) < 10: # If there are no files, break out
            return None
        self.lstRSFile = [file for file in fileList if file.find('RS') == 0]
        self.mask_exist = False
        if self.lstRSFile:
            self.lstRSFile = os.path.join(dirName,self.lstRSFile[0])
            self.check_RS_file()
            if self.mask_exist:
                fid = open(os.path.join(PathDicom,''.join(self.Contour_Names)+'.txt'),'w+')
                fid.close()
            return None
        for filename in fileList:
            try:
                ds = pydicom.read_file(os.path.join(dirName,filename))
                if ds.Modality != 'RTSTRUCT':  # check whether the file's DICOM
                    self.lstFilesDCM.append(os.path.join(dirName, filename))
                    self.Dicom_Data.append(ds)
                elif ds.Modality == 'RTSTRUCT':
                    self.lstRSFile = os.path.join(dirName, filename)
                    if self.lstRSFile and not self.mask_exist:
                        self.check_RS_file()
                        if self.mask_exist:
                            fid = open(os.path.join(PathDicom, ''.join(self.Contour_Names) + '.txt'), 'w+')
                            fid.close()
                        return None
            except:
                continue
        return None

    def check_RS_file(self):
        self.RS_struct = pydicom.read_file(self.lstRSFile)
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []
        self.rois_in_case = []
        for Structures in self.ROI_Structure:
            self.rois_in_case.append(Structures.ROIName.lower())
        # Make sure we have ALL the contours defined
        comparing = []
        for roi in self.rois_in_case:
            if roi in self.associations and self.associations[roi] in self.Contour_Names:
                self.mask_exist = True
                roi_name = self.associations[roi]
                if roi_name not in comparing:
                    comparing.append(roi_name)
                if len(comparing) == len(self.Contour_Names):
                    break
        if not set(self.Contour_Names).issubset(comparing) and not self.ignore_lacking:
            self.mask_exist = False
            for roi in self.Contour_Names:
                if roi.lower() not in self.rois_in_case:
                    print(self.PathDicom + ' lacking ' + roi)
        else:
            fid = open(os.path.join(self.PathDicom,''.join(self.Contour_Names) + '.txt'), 'w+')
            fid.close()


class Find_Image_Folders(object):
    def __init__(self, input_path = '', images_description='Images', Contour_Names=[]):
        self.Contour_Names = Contour_Names
        self.paths_to_check = []
        self.paths_done = []
        self.images_description = images_description
        self.down_folder(input_path)

    def down_folder(self,input_path):
        files = []
        dirs = []
        file = []
        for root, dirs, files in os.walk(input_path):
            break
        for val in files:
            if val.find('.dcm') != -1:
                file = val
                break
        if file:
            go = True
            if not os.path.exists(os.path.join(input_path,'made_into_nii_' + self.images_description + '.txt')) and go:
                print(input_path)
                self.paths_to_check.append(input_path)
            elif os.path.exists(os.path.join(input_path,'made_into_nii_' + self.images_description + '.txt')):
                self.paths_done.append(input_path)
        for dir in dirs:
            new_directory = os.path.join(input_path,dir)
            self.down_folder(new_directory)
        return None


class Identify_RTs_Needed:
    def __init__(self,Contour_Names = ['Liver'],Contour_Key={'Liver':1},images_description= 'Images',
                 path='S:\\SHARED\\Radiation physics\\BMAnderson\\PhD\\Liver_Ablation_Exports\\',
                 associations = None, ignore_lacking=False, thread_count=int(cpu_count()*.75-1)):
        self.ignore_lacking = ignore_lacking
        self.images_description = images_description
        self.associations = associations
        for roi in Contour_Names:
            if roi not in self.associations:
                self.associations[roi] = roi
        self.Contour_Names = Contour_Names
        self.Contour_Key = Contour_Key
        print('This is running on ' + self.Contour_Names[0] + ' contours')
        Images_Check = Find_Image_Folders(input_path=path, images_description=images_description, Contour_Names=Contour_Names)
        self.paths_to_check = Images_Check.paths_to_check
        print('This is running on ' + str(thread_count) + ' threads')
        q = Queue(maxsize=thread_count)
        A = [q,self.associations, Contour_Names, ignore_lacking]
        threads = []
        for worker in range(thread_count):
            t = Thread(target=worker_def, args=(A,))
            t.start()
            threads.append(t)
        for path in self.paths_to_check:
            if not os.path.exists(os.path.join(path, ''.join(self.Contour_Names) + '.txt')):
                q.put(path)
        for i in range(thread_count):
            q.put(None)
        for t in threads:
            t.join()
        self.paths_to_check += Images_Check.paths_done


class Find_Contour_Files(object):
    def __init__(self, Contour_Names=[], check_paths = [''],images_description=''):
        self.paths_to_check = {}
        self.temp_paths_to_check = []
        iterations = []
        self.i = 0
        self.Contour_Names = Contour_Names
        for path in check_paths:
            files = []
            for _, _, files in os.walk(path):
                break
            iteration_files = [i for i in files if i.find('Iteration_') != -1]
            include = True
            if iteration_files:
                for i in iteration_files:
                    if i.split('_Iteration_')[0] == images_description:
                        iteration = int(i.split('Iteration_')[1].split('.txt')[0])
                        iterations.append(iteration)
                        include = False
                        if 'made_into_nii_' + images_description + '.txt' not in files:
                            self.paths_to_check[path] = iteration
                        break
            if include:
                if ''.join(self.Contour_Names) + '.txt' in files:
                    self.temp_paths_to_check.append(path)
        for path in self.temp_paths_to_check:
            while self.i in iterations:
                self.i += 1
            self.paths_to_check[path] = self.i
            iterations.append(self.i)

class DicomImagestoData:
    image_size = 512
    def __init__(self,Contour_Names = ['Liver'],Contour_Key={'Liver':1},path='S:\\SHARED\\Radiation physics\\BMAnderson\\PhD\\Liver_Ablation_Exports\\',
                 data_path='\\\\mymdafiles\\di_data1\\Morfeus\\bmanderson\\CNN\\Cervical_Data\\',
                 images_description= 'Images',associations=None, argmax=True):
        self.argmax = argmax
        self.guiding_exams = {}
        self.reader = sitk.ImageSeriesReader()
        self.reader.MetaDataDictionaryArrayUpdateOn()
        self.reader.LoadPrivateTagsOn()
        self.data_path = data_path
        self.associations = associations
        for roi in Contour_Names:
            if roi not in self.associations:
                self.associations[roi] = roi
        self.images_description = images_description
        self.Contour_Names = Contour_Names
        self.Contour_Key = Contour_Key
        print('This is running on ' + self.Contour_Names[0] + ' contours')
        self.MRN_list = os.listdir(path)
        self.got_file_list = False
        self.iteration = 0
        self.hierarchy = {'liver':['liver_ethan pv','liver','liver_bma_program_4']}
        for key in self.hierarchy:
            for value in self.hierarchy[key]:
                self.associations[value] = key
        self.iteration = 0
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        self.batch_size = 1
        self.perc_done = 0
        self.patient_spacing_info = load_obj(self.data_path.split('Numpy')[0] + 'patient_info_' + self.images_description + '.pkl')
        if not self.patient_spacing_info:
            self.patient_spacing_info = {images_description:{'path':data_path}}


    def Make_Contour_From_directory(self,A):
        PathDicom, iteration = A
        self.iteration = iteration
        self.prep_data(PathDicom)
        self.get_images()
        self.get_mask()
        self.mask_array_and_mask()
        print('iteration ' + str(self.iteration) + ' completed')
        fid = open(os.path.join(PathDicom,'made_into_nii_' + self.images_description + '.txt'),'w+')
        fid.close()
        fid = open(os.path.join(PathDicom,self.images_description + '_Iteration_' + str(self.iteration) + '.txt'),'w+')
        fid.close()
        self.iteration += 1
        return None


    def prep_data(self,PathDicom):
        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.lstRSFile = []
        self.Dicom_Data = []
        fileList = []
        for dirName, dirs, fileList in os.walk(PathDicom):
            break
        fileList = [i for i in fileList if i.find('.dcm') != -1]
        if len(fileList) < 10: # If there are no files, break out
            return None
        self.dicom_names = self.reader.GetGDCMSeriesFileNames(self.PathDicom)
        self.ds = pydicom.read_file(self.dicom_names[0])
        self.reader.SetFileNames(self.dicom_names)
        image_files = [i.split(PathDicom)[1][1:] for i in self.dicom_names]
        self.lstRSFiles = [file for file in fileList if file not in image_files]
        if os.path.exists(dirName + 'new_RT_renamed.dcm'):
            self.lstRSFile = dirName + 'new_RT_renamed.dcm'
        self.RS_struct = pydicom.read_file(os.path.join(PathDicom,self.lstRSFiles[0]))
        if Tag((0x3006, 0x020)) in self.RS_struct.keys():
            self.ROI_Structure = self.RS_struct.StructureSetROISequence
        else:
            self.ROI_Structure = []

    def mask_array_and_mask(self):
        self.perc_done += 1
        try:
            SeriesDescription = str(self.ds.SeriesDescription)
            if SeriesDescription.find('CT') != -1 or self.ds.Modality == 'CT':
                add_info = 'CT'
                add = add_info + '\\'
            elif SeriesDescription.find('MR') != -1 or self.ds.Modality == 'MR':
                add_info = 'MR'
                if os.path.exists(os.path.join(self.PathDicom,'SeriesDescription.txt')):
                    fid = open(os.path.join(self.PathDicom,'SeriesDescription.txt'))
                    SeriesDescription = fid.readline()
                    fid.close()
                if SeriesDescription.find('T1') != -1:
                    if SeriesDescription.find('+Gd') != -1:
                        add_info += '_T1+Gd'
                    else:
                        add_info += '_T1'
                elif SeriesDescription.lower().find('flair') != -1:
                    add_info += '_FLAIR'
                elif SeriesDescription.find('T2') != -1:
                    add_info += '_T2'
                else:
                    add_info += '_other'
                add_info = 'MR_other'
                add = add_info + '\\'
            else:
                add_info = 'other'
                add = add_info + '\\'
            desc = self.ds.SeriesDescription
        except:
            add_info = 'other'
            add = add_info + '\\'
            desc = 'Error...'
            print('--- error with series description..')
        if not self.got_file_list:
            self.get_files_in_output_dirs([add_info])
        self.iterations['all_vals'].append(self.iteration)
        if not os.path.exists(os.path.join(self.data_path,add)):
            os.makedirs(os.path.join(self.data_path,add))
        image_path = os.path.join(self.data_path, add, 'Overall_Data_' + self.images_description + '_' + add_info + '_' + str(self.iteration) + '.nii.gz')
        sitk.WriteImage(self.dicom_images,image_path)
        dtype = 'int8'
        annotation_path = os.path.join(self.data_path, add, 'Overall_mask_' + self.images_description + '_' + add_info + '_y' + str(self.iteration) + '.nii.gz')
        new_annotation = sitk.GetImageFromArray(self.mask.astype(dtype))
        new_annotation.SetSpacing(self.dicom_images.GetSpacing())
        new_annotation.SetOrigin(self.dicom_images.GetOrigin())
        new_annotation.SetDirection(self.dicom_images.GetDirection())
        sitk.WriteImage(new_annotation,annotation_path)
        fid = open(os.path.join(self.data_path, add,
                                self.images_description + '_Iteration_' + str(self.iteration) + '.txt'), 'w+')
        fid.write(str(self.ds.PatientID) + ',' + str(self.ds.SliceThickness) + ',' + str(self.ds.PixelSpacing[0]) + ',' + desc)
        fid.close()
        return None

    def get_files_in_output_dirs(self,dirs):
        iterations = {}
        all_vals = []
        for dir_val in dirs:
            iterations[dir_val] = []
            if not os.path.exists(os.path.join(self.data_path,dir_val)):
                os.makedirs(os.path.join(self.data_path,dir_val))
            for file in os.listdir(os.path.join(self.data_path,dir_val)):
                if file.find('Overall_Data') == 0:
                    file = file.split(self.images_description)[1]
                    iteration = file.split('_')[-1].split('.')[0]
                    iterations[dir_val].append(int(iteration))
                    all_vals.append(int(iteration))
            iterations[dir_val].sort()
        all_vals.sort()
        self.iterations = iterations
        self.iterations['all_vals'] = all_vals
        self.got_file_list = True

    def get_images(self):
        self.dicom_images = self.reader.Execute()
        # slice_location_key = "0020|1041"
        sop_instance_UID_key = "0008|0018"
        self.SOPInstanceUIDs = [self.reader.GetMetaData(i,sop_instance_UID_key) for i in range(self.dicom_images.GetDepth())]
        self.image_size_1, self.image_size_2, self.num_images = self.dicom_images.GetSize()

    def get_mask(self):
        self.mask = np.zeros([self.num_images,self.image_size_1, self.image_size_2, len(self.Contour_Names)+1],
                             dtype='int8')

        self.structure_references = {}
        for contour_number in range(len(self.RS_struct.ROIContourSequence)):
            self.structure_references[self.RS_struct.ROIContourSequence[contour_number].ReferencedROINumber] = contour_number
        found_rois = {}
        for roi in self.Contour_Names:
            found_rois[roi] = {'Hierarchy':999,'Name':[],'Roi_Number':0}
        for Structures in self.ROI_Structure:
            ROI_Name = Structures.ROIName.lower()
            if Structures.ROINumber not in self.structure_references.keys():
                continue
            true_name = None
            if ROI_Name in self.associations:
                true_name = self.associations[ROI_Name]
            elif ROI_Name.lower() in self.associations:
                true_name = self.associations[ROI_Name.lower()]
            if true_name and true_name in self.Contour_Names:
                if true_name in self.hierarchy.keys():
                    for roi in self.hierarchy[true_name]:
                        if roi == ROI_Name:
                            index_val = self.hierarchy[true_name].index(roi)
                            if index_val < found_rois[true_name]['Hierarchy']:
                                found_rois[true_name]['Hierarchy'] = index_val
                                found_rois[true_name]['Name'] = ROI_Name
                                found_rois[true_name]['Roi_Number'] = Structures.ROINumber
                else:
                    found_rois[true_name] = {'Hierarchy':999,'Name':ROI_Name,'Roi_Number':Structures.ROINumber}
        for ROI_Name in found_rois.keys():
            if found_rois[ROI_Name]['Roi_Number'] in self.structure_references:
                index = self.structure_references[found_rois[ROI_Name]['Roi_Number']]
                mask = self.get_mask_for_contour(index)
                self.mask[...,self.Contour_Names.index(ROI_Name)+1][mask == 1] = 1
        if self.argmax:
            self.mask = np.argmax(self.mask,axis=-1)
        return None

    def get_mask_for_contour(self,i):
        self.Liver_Locations = self.RS_struct.ROIContourSequence[i].ContourSequence
        return self.Contours_to_mask()

    def Contours_to_mask(self):
        mask = np.zeros([self.num_images, self.image_size_1, self.image_size_2], dtype='int8')
        Contour_data = self.Liver_Locations
        ShiftCols, ShiftRows, _ = [float(i) for i in self.reader.GetMetaData(0,"0020|0032").split('\\')]
        # ShiftCols = self.ds.ImagePositionPatient[0]
        # ShiftRows = self.ds.ImagePositionPatient[1]
        PixelSize = self.dicom_images.GetSpacing()[0]
        Mag = 1 / PixelSize
        mult1 = mult2 = 1
        if ShiftCols > 0:
            mult1 = -1
        if ShiftRows > 0:
            print('take a look at this one...')
        #    mult2 = -1

        for i in range(len(Contour_data)):
            referenced_sop_instance_uid = Contour_data[i].ContourImageSequence[0].ReferencedSOPInstanceUID
            if referenced_sop_instance_uid not in self.SOPInstanceUIDs:
                print('Error here with instance UID')
                return None
            else:
                slice_index = self.SOPInstanceUIDs.index(referenced_sop_instance_uid)
            cols = Contour_data[i].ContourData[1::3]
            rows = Contour_data[i].ContourData[0::3]
            col_val = [Mag * abs(x - mult1 * ShiftRows) for x in cols]
            row_val = [Mag * abs(x - mult2 * ShiftCols) for x in rows]
            temp_mask = self.poly2mask(col_val, row_val, [self.image_size_1, self.image_size_2])
            mask[slice_index][temp_mask > 0] = 1
            #scm.imsave('C:\\Users\\bmanderson\\desktop\\images\\mask_'+str(i)+'.png',mask_slice)

        return mask

    def poly2mask(self,vertex_row_coords, vertex_col_coords, shape):
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[fill_row_coords, fill_col_coords] = True
        return mask

def main(image_path=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Pancreas\Pancreas\Koay_patients\Images',ignore_lacking=False,
         out_path=r'K:\Morfeus\BMAnderson\CNN\Data\Data_Pancreas\Pancreas\Koay_patients\Numpy', images_description='',
         Contour_Names=['gtv','ablation'],associations=None,argmax=True, thread_count = int(cpu_count()*0.75-1)):
    '''
    :param image_path: Path to the image files
    :param ignore_lacking: Ignore when a structure lacks all necessary contours, experimental
    :param out_path: Path to output folder
    :param images_description: Description of images
    :param Contour_Names: list of contour names desired
    :param associations: a dictionary of ROI_Name: Desired_ROI_Name
    :param thread_count: the number of threads to use
    :param argmax: if the mask should be collapsed based on np.argmax()
    :return:
    '''
    associations = correct_association_file(associations) # Make the keys and results lower-case
    Contour_Names = [i.lower() for i in Contour_Names]
    Contour_Key = {}
    for i, name in enumerate(Contour_Names):
        Contour_Key[name] = i + 1
    start_pat = 0
    k = Identify_RTs_Needed(Contour_Names=Contour_Names, Contour_Key=Contour_Key,
                      path=image_path,images_description=images_description,ignore_lacking=ignore_lacking,
                      associations=associations, thread_count=thread_count)
    Folders_w_Contours = Find_Contour_Files(Contour_Names=Contour_Names, check_paths=k.paths_to_check).paths_to_check

    print('This is running on ' + str(thread_count) + ' threads')
    q = Queue(maxsize=thread_count)
    A = [q, Contour_Names, Contour_Key, image_path, out_path, images_description, associations, argmax]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def_write, args=(A,))
        t.start()
        threads.append(t)
    for path in Folders_w_Contours:
        print(path)
        q.put([path,Folders_w_Contours[path]+start_pat])
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()
    main_run(images_description=images_description,base_path=out_path)
    make_location_pickle(out_path,image_path,images_description)
    separate(desc=images_description, path_base=out_path)
'''
Run this one, then do 
Make_Patient_pickle_file_from_text.py
then do
Separate_Numpy_Images_Into_Test_Train_Validation.py
'''
if __name__ == '__main__':
    xxx = 1
    # base_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Liver_Disease_Ablation_Segmentation'
    # images_description = 'Disease_Ablation'
    # out_path = os.path.join(base_path,'Numpy_' + images_description)
    # main(base_path=base_path,image_path=os.path.join(base_path,'Images'),out_path=out_path,
    #      images_description=images_description, Contour_Names=['Liver','GTV','Ablation'])