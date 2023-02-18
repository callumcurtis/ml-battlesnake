ROOTDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

compile: clean
	cd $(ROOTDIR)/rules; go build -o $(ROOTDIR)/bin/rules.dll -buildmode=c-shared .
	cd $(ROOTDIR)/engine/cli/battlesnake; go build -o $(ROOTDIR)/bin/engine

clean:
	rm -rf $(ROOTDIR)/bin/**
