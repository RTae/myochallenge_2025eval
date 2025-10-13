
train:
	docker run -d --name myochallenge_train \
		--gpus all \
		--shm-size=30gb \
		-v ./logs:/app/logs \
		ghcr.io/rtae/myochallenge/myochallenge-train:latest
	
stop:
	docker stop myochallenge_train

delete:
	docker rm -f myochallenge_train

log:
	docker logs -f myochallenge_train