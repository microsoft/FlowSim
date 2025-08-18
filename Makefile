IMAGE_NAME     ?= workloadsim-image
CONTAINER_NAME ?= workloadsim-docker
PORT           ?= 22401
MEM_LIMIT      ?= 240g
MOUNT_VOLUME   ?= n

fix-permissions:
	sudo chown -R $$(id -u):$$(id -g) $$(pwd)

build-docker:
	sudo docker build -t $(IMAGE_NAME) -f dockerfiles/cuda12.6.dockerfile .;

run-docker:
	@if [ -n "$(GPU_DEVICES)" ]; then GPU_DEVICES_FLAG="--gpus=$(GPU_DEVICES)"; else GPU_DEVICES_FLAG=""; fi; \
	if [ "$(MOUNT_VOLUME)" = "y" ]; then VOLUME_FLAG="-v $$(pwd):/workloadsim"; else VOLUME_FLAG=""; fi; \
	echo "Running: sudo docker run $$VOLUME_FLAG $$GPU_DEVICES_FLAG --network=host --cap-add=SYS_ADMIN -it -p ${PORT}:22 --memory ${MEM_LIMIT} --name $(CONTAINER_NAME) $(IMAGE_NAME)"; \
	sudo docker run $$VOLUME_FLAG $$GPU_DEVICES_FLAG --cap-add=SYS_ADMIN --network=host --shm-size 911G -it -p ${PORT}:22 --memory ${MEM_LIMIT} --name $(CONTAINER_NAME) $(IMAGE_NAME)

rm-docker:
	sudo docker rm $(CONTAINER_NAME)