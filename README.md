## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!
# This code feeds into https://github.com/brianmanderson/Make_Single_Images
This is for the creation of numpy arrays from dicom images and RT structures for deep learning purposes

The DicomImagesintoData_Parallel should be able to create numpy arrays with a training/test/validation split

You will need to define the contour names that you want to create


    from DicomImagesintoData_Parallel import main
    Contour_Names = ['Liver']
    path = '\\\\server\\location\\Liver_Patients\\Patient_Images\\'
    # Where the location above has folders with patient images. The RT structures and images must be in the same folder!
    out_path = '\\\\server\\location\\Liver_Patients\\Numpy_Arrays\\'
    images_description = 'My_Liver_Images'
    main(Contour_Names,path,out_path,images_description)
    # The output should be '\\server\\location\\Liver_Patients\\Numpy_Arrays\\My_Liver_Images\\(Train\Test\Validation)'

