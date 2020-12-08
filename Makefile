docs: ## Render annotated HTML
	pylit --remove_empty_sections --title_md -t ../../pylit/templates/default -d html -w ppo.py

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: docs
.DEFAULT_GOAL := help
