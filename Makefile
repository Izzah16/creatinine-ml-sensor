# One-command setup for training + running GUI app

install:
	pip install -r requirements.txt

train:
	python train_model.py

run:
	python mainapp.py
