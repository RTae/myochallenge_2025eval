
train:
	docker compose -p myochallenge up -d
	
stop:
	docker compose -p myochallenge down

log:
	docker compose -p myochallenge logs -f