docker build --platform linux/amd64 -t dataset_cheetah .

docker run --platform linux/amd64 -it --rm --name dataset_cheetah_v1 dataset_cheetah