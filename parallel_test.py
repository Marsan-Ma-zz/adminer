#-*- coding:utf-8 -*-

import time
import pickle

from cjobs import add
from cjobs import train_model_worker

# # test
# res = add.delay(3, 8)
# while not res.ready(): 
#   print 'waiting simple'
#   time.sleep(1)
# print res.get()

# # train one model 
# res = train_model_worker.delay('ocpm', 'conv_page_engagement', industry='EC')
# while not res.ready(): 
#   print 'waiting'
#   time.sleep(1)

# best_gs_svr, scaler, filename = res.get()
# pickle.dump([best_gs_svr, scaler], open("./result/"+filename, "w"))
# print "%s written." % filename

