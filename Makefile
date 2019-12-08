# developer Makefile for repeated tasks
# 
.PHONY: clean

nb:
	docker run -it --rm  -p 8888:8888 -v ${PWD}:/home/jovyan sjsrey/pysaldev:2.3 sh -c "/home/jovyan/develop.sh && /bin/bash"

term:
	docker run -it --rm   -v ${PWD}:/home/jovyan sjsrey/pysaldev:2.3 sh -c "/home/jovyan/develop.sh && /bin/bash"

download:
	python gitreleases.py

convert:
	python convert.py

test:
	nosetests 

doctest:
	cd doc; make pickle; make doctest

install:
	python setup.py install >/dev/null

src:
	python setup.py sdist >/dev/null

win:
	python setup.py bdist_wininst >/dev/null

prep:
	rm -rf pysal/lib
	mkdir pysal/lib

docs:
	python convert_docs.py
	cd doc; make clean; make html

clean: 
	find . -name "*.pyc" -exec rm '{}' ';'
	find pysal -name "__pycache__" -exec rm -rf '{}' ';'
	rm -rf dist
	rm -rf build
	rm -rf PySAL.egg-info
	rm -rf tmp

