.PHONY: lint test

MAKE_LINT_ERROR_OUT ?= 0
ISORT_FLAGS := $(if $(filter 1,$(MAKE_LINT_ERROR_OUT)),--check-only,)
BLACK_FLAGS := $(if $(filter 1,$(MAKE_LINT_ERROR_OUT)),--check,)

lint:
	isort $(ISORT_FLAGS) .
	black $(BLACK_FLAGS) .
	flake8 .

test:
	cd tests && pytest .
