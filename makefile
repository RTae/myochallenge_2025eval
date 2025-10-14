build:
	docker build \
	-t ghcr.io/rtae/myochallenge/myochallenge-train:latest \
	-f Dockerfile.train .

train:
	docker compose -p myochallenge up -d
	
stop:
	docker compose -p myochallenge down

log:
	docker compose -p myochallenge logs -f

clean_exp:
	sudo rm -rf ./logs/tabletennis_*