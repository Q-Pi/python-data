FROM ubuntu

WORKDIR workspace
COPY . .

RUN apt-get -y update 						&& \
	apt-get -y install python3  			&& \
	apt-get -y install python3-pip  		&& \
	apt-get -y install vim 					&& \
	apt-get -y install git 					&& \
	git config --global user.name fakeaccount-learning && \
	git config --global user.email kentin91330@gmail.com && \
	git clone https://github.com/fakeaccount-learning/fakerepo.git 	&& \
	python3 -m pip install --upgrade pip  	&& \
	python3 -m pip install -r requirements.txt

EXPOSE 8000

CMD ["jupyter", "notebook", "--allow-root", "--port=8000", "--ip=0.0.0.0", "--no-browser"]