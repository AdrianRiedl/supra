all:
ifeq (, $(shell which latexmk))
	$(error "No latexmk in $(PATH), consider doing apt-get install latexmk")
endif
	latexmk -bibtex projectdescription.tex -pdf
	latexmk -c