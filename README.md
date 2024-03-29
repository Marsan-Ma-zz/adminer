# Adminer

It's a trimmed version of ad campaign analysis platform.

### Why python ?
  1. need NLTK and those libs for NLP.
  2. need traverse through all existing articles, thus need keep big database.
     since its no sense to duplicate same db in ruby, so python also take web.
  3. NLP needs to keep dict, corpus, lsi models in RAM, thus a live python process is needed.
  4. while keep playing with data in future, python is the best choice.

### Python Virtual Env
  1. install    : sudo easy_install virtualenv
  2. create     : virtualenv my_pyenv
  3. activate   : source my_pyenv/bin/activate
  4. deactivate : deactivate

### Deploy: Bottle + CherryPy + Supervisord
  # CherryPy + Bottle
  bottle.run(server='cherrypy', host='api.piposay.com', port=9800)  # in python main file 

  # Supervoid : http://supervisord.org/running.html
  1. install    : pip install supervisor
  2. configue   : echo_supervisord_conf > supervisord.conf , then sudo mv supervisord.conf /etc/supervisord.conf  
  3. add program in /etc/supervisord.conf
  4. run        : supervisord
  5. restart    : just 'kill -9 <pid>' in shell
  6. startup    : get /etc/init.d/supervisord from https://github.com/Supervisor/initscripts

### Run
  # if file 'tmp/is_dev' exists, start as dev-mode
  ipython piposay # cherrypy server for bottle

  # contents
  + piposay.py    : main program, page extract & summary, with bottle + cherrypy web server.
  + collector.py  : scrape whole-site content from sitemap.
  + gravity.py    : build LSA model from corpus, find similar topics.

  /usr/bin/ipython notebook --profile=myserver  # start remote ipython-notebook
  sudo supervisorctl restart piposay # restart job, see /etc/supervisord.conf

### Light ORM for MongoDB : Ming
  print schema.Article.m.find().count()     # calculate count
  post1 = schema.Article.m.find({'title': 'MyPage'}).all()[3]  # fetch one
  print post1.title

  post2 = schema.Article.m.get(title='MyPage')[3] # same as last

  post3 = schema.Article(dict(title='MyPage', text='')) # new post 
  post3.m.save()  # save it

### Chinese terms separation, keywords extraction
  [Jieba]
    https://github.com/fxsjy/jieba
    alg: http://ddtcms.com/blog/archive/2013/2/4/69/jieba-fenci-suanfa-lijie/

### Extract real content from html
  [JustText] https://github.com/miso-belica/jusText
    alg: http://code.google.com/p/justext/wiki/Algorithm)

  [Python-Goose] https://github.com/xgdlm/python-goose by GravityLab
    goose demo: http://jimplush.com/blog/goose


### Other resources
  [Gensim] topic modeling, similarity query.
    http://radimrehurek.com/gensim/

  [Articles Categorize using NaiveBayesClassifier]
    http://www.ibm.com/developerworks/cn/opensource/os-pythonnltk/#list4

  [Ming for MongoDB]
    http://merciless.sourceforge.net/tour.html
