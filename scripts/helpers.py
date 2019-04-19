import numpy as np

def file_csv_to_txt(file_name):
    file_name = list(file_name)
    file_name[-1] = 't'
    file_name[-2] = 'x'
    file_name[-3] = 't'
    return ''.join(file_name)

def file_csv_to_json(file_name):
    file_name = list(file_name)
    file_name[-1] = 'o'
    file_name[-2] = 's'
    file_name[-3] = 'j'
    file_name = ''.join(file_name)
    file_name = file_name + "n"
    return file_name

def get_meta_file_name(file_name, meta_tag):
    return_name = file_name[:-4] + "_" + meta_tag + ".txt"
    return return_name

def get_result_file_name(file_name, results_dir,model):
    file_name = file_name.split("/")
    file_name = results_dir + "/data/"+model+"/" + file_name[-1]
    return file_name

def get_result_dump_name(file_name):
    file_name = file_name.split("/")
    return file_name[-1]

def dump_results(results_array, results_dir, model):
    dump_file = open(results_dir+"/"+model+"_list.csv",'w')
    for r in results_array:
        dump_file.write(r + ",\n")
    dump_file.close()

def dump_files_with_nan(nan_file_array, results_dir, model):
    dump_file = open(results_dir+"/"+model+"_nan_list.csv",'w')
    for r in nan_file_array:
        dump_file.write(r + ",\n")
    dump_file.close()

def remove_tag_from_file(file_name):
    file_name = file_name.split("/")
    return_name = ""
    for i in range(0,len(file_name)-1):
        return_name = return_name + file_name[i] + "/"
    
    last_name = file_name[-1].split("_",1)
    return_name = return_name + last_name[-1]
    return return_name

def check_nan(np_array):
    for e in np.isnan(np_array):
        if e == True:
            return True
    return False

def MSE(y, y_hat):
    return np.around(np.square(y - y_hat).mean(), decimals=6)