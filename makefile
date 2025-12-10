build:
	sudo docker build \
		-t ghcr.io/rtae/myochallenge/myochallenge-train:latest \
		-f Dockerfile.train .

train:
	sudo docker compose -p myochallenge up -d
	
stop:
	sudo docker compose -p myochallenge down

log:
	sudo docker compose -p myochallenge logs -f

clean_exp:
	sudo rm -rf ./logs