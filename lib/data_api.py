#-*- coding:utf-8 -*-

# basic
import os, timeit, copy, skimage
import requests, urllib2
from skimage import io
from skimage import transform as img_tran


#===========================================
#   Image Recognition
#===========================================
def img_auto_tag(path, adgroup=None, tag_num=10):
  files = {'source': urllib2.urlopen(path).read()} if 'http' in path else {'source': open(path, 'rb')}
  result = requests.post(url, files=files, data={'top':tag_num, 'model':'image-net-2012-vgg-d'})
  # print "\n".join(map(lambda r: "%s(%.3f)" % (r['word'], r['confidence']), result.json()))
  try:
    tags = map(lambda r: [r['word'], r['confidence']], result.json())
  except:
    print "[Error] for result=%s" % result
    adgroup.update(set__cr_image_tags=[])
    return None
  if adgroup:
    adgroup.update(set__cr_image_tags=tags)
    # print "update %s as %s" % (adgroup.cr_image_url, tags)
  else:
    return tags


def img_find_file(adgroup_id):
  for n in ['png', 'jpeg']: #, 'gif']:
    fn = './images/%s.%s' % (adgroup_id, n)
    if os.path.isfile(fn): return fn
  return None

def img_show(filename):
  img = io.imread(filename)
  print img.shape, filename
  io.imshow(img)
  io.show()
  return img.shape

