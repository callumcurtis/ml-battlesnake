ROOTDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

compile: clean
	cd $(ROOTDIR)/ml_battlesnake/learning/game-engine-as-dll; go build -o $(ROOTDIR)/bin/engine.dll -buildmode=c-shared .
	cd $(ROOTDIR)/ml_battlesnake/deployment/game-engine-as-executable/cli/battlesnake; go build -o $(ROOTDIR)/bin/engine

clean:
	rm -rf $(ROOTDIR)/bin/**
