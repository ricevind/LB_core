# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:23:05 2016

@author: ricevind
"""
import os
import datetime
import shutil

def arch_create(name, geo):
    geometries = {'ane':'Aneurysm', 'bif':'Bifurcation', 'aor':'Aorta'}
    
    if geo not in geometries:
      return 'Wrong geometry folder'
    release = '/home/ricevind/Documents/Praca MGR/Release'
    
    folder = os.path.join(release,geometries[geo], name)
    try:
      os.mkdir(folder)
      os.mkdir(os.path.join(folder, 'Zdjecia'))
    except:
      print('prawdopodobnie już istnieje')
      
    return folder
    
  
def arch_param(folder, params, paramslist):
    with open(os.path.join(folder, 'params'), 'a') as file:
      newline = "######################## {} ########################## \n".format(datetime.datetime.now())
      file.write(newline)
      for name in paramslist:
        file.write(name+' = '+str(params[name])+'\n')


def create_proba(glowny):
    dir_skrypt = r'/home/ricevind/Documents/Praca MGR/Produkcja'
    dir_glowny = os.path.join(os.path.split(dir_skrypt)[0], glowny)
    try:    
        os.mkdir(dir_glowny)
    except:
        print('prawdopodobnie już istnieje taki branch sumulacji lub kolejna proba')
    return dir_glowny
        
def create_run(dir_glowny, rewrite=None, test=None):
     
    dirs = next(os.walk(dir_glowny))[1]
    numbers = [int(fold[-3:]) for fold in dirs]
    try:
        n = max(numbers)
    except:
        n = 0
    n =n+ 1
    name = 'run{0:03d}'.format(n)
    if rewrite:
        erase_path = os.path.join(dir_glowny, test)
        shutil.rmtree(erase_path)
        name = test
    test_path = os.path.join(dir_glowny,name)
    os.mkdir(test_path)
    zdjecia_path = os.path.join(test_path, 'Zdjecia')
    os.mkdir(zdjecia_path)
    obliczenia_path = os.path.join(test_path, 'Obliczenia')
    os.mkdir(obliczenia_path)
    return test_path, zdjecia_path, obliczenia_path, name

def codesafe(path, dir_skrypt, nazwa):
    path = os.path.join(path,'kod.py')
    dir_skrypt =  os.path.join( os.path.split(dir_skrypt)[0],nazwa)
    shutil.copyfile(dir_skrypt, path)
