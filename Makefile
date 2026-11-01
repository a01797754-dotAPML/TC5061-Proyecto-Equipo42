#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici-mlops
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m mibici_mlops.dataset
	$(PYTHON_INTERPRETER) -m mibici_mlops.manifest

## Validate data with DeepChecks (run inside the mibici-dc environment)
.PHONY: deepchecks
deepchecks:
	$(PYTHON_INTERPRETER) -m mibici_mlops.validacion
	$(PYTHON_INTERPRETER) -m pytest tests/test_validacion.py

## Run security checks (bandit, pip-audit) from an environment with requirements-security.txt
.PHONY: security
security:
	$(PYTHON_INTERPRETER) -m bandit -r mibici_mlops -ll -q
	$(PYTHON_INTERPRETER) -m pip_audit -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip_audit -r requirements-deepchecks.txt --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w

## Run baseline experiments (P1, P2, P3) and log them to MLflow
.PHONY: experiments
experiments:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.run_experiments
	
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
