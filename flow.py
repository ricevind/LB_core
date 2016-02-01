# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:38:25 2015

@author: ricevind
"""
import os
import shutil
def animate(path, filename, pathd, framerate=10):

    file = os.path.join(pathd, filename+".mpg")
    os.chdir(path)

    os.system("mencoder 'mf://*.png' -mf type=png:fps={} -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o ".format(framerate) + filename )
    shutil.move(os.path.join(path, filename), file)
#for i, j, k in os.walk(r'/home/ricevind/Documents/Praca MGR/Zdjecia'):
#     for k in k:
#         if len(k) == 12:
#             os.rename(k, '0'+k[4:6]+'.png')
#         if len(k) == 13:
#             os.rename(k, k[4:7]+'.png')
#         else:
#             continue