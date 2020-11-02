FROM ubuntu

COPY . /

RUN apt-get -y update 						&& \
	apt-get -y install python3  			&& \
	apt-get -y install python3-pip  		&& \
	apt-get -y install vim 					&& \
	apt-get -y install git 					&& \
	git clone https://github.com/fakeaccount-learning/fakerepo.git 	&& \
	python3 -m pip install --upgrade pip  	&& \
	python3 -m pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["jupyter-notebook", "--allow-root"]