
train:
	docker run -d --name myochallenge_train \
		--gpus all \
		--shm-size=30gb \
		-v ./logs:/app/logs \
		-e MUJOCO_GL=egl \
		-e EGL_DEVICE_ID=0 \
		ghcr.io/rtae/myochallenge/myochallenge-train:latest
	
stop:
	docker stop myochallenge_train

delete:
	docker rm -f myochallenge_train

log:
	docker logs -f myochallenge_train