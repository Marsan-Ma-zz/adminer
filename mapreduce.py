#-*- coding:utf-8 -*-

from lib import schema as db
from mongoengine import *
from pymongo import MongoClient
# connect('adminer', host='mldm-mongo.azure.funptw', port=12999)

#=================================================
#   Map Reduce with MongoEngine
#=================================================
def test_map_reduce():
  """Ensure map/reduce is both mapping and reducing.
  """
  class BlogPost(Document):
      title = StringField()
      tags = ListField(StringField(), db_field='post-tag-list')

  BlogPost.drop_collection()
  BlogPost(title="Post #1", tags=['music', 'film', 'print']).save()
  BlogPost(title="Post #2", tags=['music', 'film']).save()
  BlogPost(title="Post #3", tags=['film', 'photography']).save()
  map_f = """
      function() {
          this[~tags].forEach(function(tag) {
              emit(tag, 1);
          });
      }
  """
  reduce_f = """
      function(key, values) {
          var total = 0;
          for(var i=0; i<values.length; i++) {
              total += values[i];
          }
          return total;
      }
  """
  # run a map/reduce operation spanning all posts
  results = BlogPost.objects.map_reduce(map_f, reduce_f, "myresults")
  for r in results:
    print r.key + ' / ' + str(r.value)
    # print r.value
  BlogPost.drop_collection()


#=================================================
#   Map-Reduce with PyMongo
#=================================================
def merge_performance():
  db = MongoClient('mldm-mongo.azure.funptw', port=12999).adminer
  mapper = Code('''
    function () {
      emit(
        this.adgroup_id, this.spent
      );
    }
  ''')
  reducer = Code('''
    function(key, values) {
      var result = { 
        spent: 0,
        lifetime_unique_impressions: 0, 
        lifetime_unique_clicks: 0,
      };
      for(var i=0; i<values.length; i++) {
        result.spent += values[i];
        result.lifetime_unique_impressions += 200;
        result.lifetime_unique_clicks += 300;
      }
      return result; 
    }
  ''')

  results = db.adgroup_dailys.map_reduce(mapper, reducer, 'inline', limit=30)
  for doc in results.find():
    print doc


#=================================================
#   Main
#=================================================
test_map_reduce()
# merge_performance()
