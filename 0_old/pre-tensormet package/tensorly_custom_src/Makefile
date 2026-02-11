# Automate testing etc
BACKEND?='numpy'

.PHONY: all install debug test test-all test-coverage

all: install test

install:
	pip install -e .

debug:
	TENSORLY_BACKEND=$(BACKEND) pytest -v --pdb tensorly_custom

test:
	TENSORLY_BACKEND=$(BACKEND) pytest -v tensorly_custom

test-all:
	TENSORLY_BACKEND='numpy' pytest -v tensorly_custom
	TENSORLY_BACKEND='cupy' pytest -v tensorly_custom
	TENSORLY_BACKEND='pytorch' pytest -v tensorly_custom
	TENSORLY_BACKEND='jax' pytest -v tensorly_custom
	TENSORLY_BACKEND='tensorflow' pytest -v tensorly_custom

test-coverage:
	TENSORLY_BACKEND=$(BACKEND) pytest -v --cov tensorly_custom tensorly_custom

