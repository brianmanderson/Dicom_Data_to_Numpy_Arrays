from Utils import os, save_obj

def get_iteration_location(input_path, Description='', output={}):
    files = []
    dirs = []
    for root, dirs, files in os.walk(input_path):
        break
    files = [os.path.join(input_path,i) for i in files if i.find(Description + '_Iteration_') == 0]
    if files:
        iteration = (files[0].split('Iteration_')[-1]).split('.txt')[0]
        output[iteration] = input_path
    for dir_val in dirs:
        output = get_iteration_location(os.path.join(input_path,dir_val),Description, output)
    return output

def make_location_pickle(base_path, path, Description):
    output = get_iteration_location(path, Description=Description)
    key_list = list(output.keys())
    for key in key_list:
        output[key] = output[key].split(path)[-1].split('\\')[-1]
    save_obj(os.path.join(base_path, 'Data_Locations.pkl'), output)
    return None

if __name__ == '__main__':
    xxx = 1