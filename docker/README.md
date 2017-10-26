[![Docker Automated buil](https://img.shields.io/docker/automated/enriquegrodrigo/docker-sparkdev.svg)](https://hub.docker.com/r/enriquegrodrigo/docker-sparkdev/)
[![Docker Build Statu](https://img.shields.io/docker/build/enriquegrodrigo/docker-sparkdev.svg)](https://hub.docker.com/r/enriquegrodrigo/docker-sparkdev/)
# docker-sparkdev

This is a base base image for developing Apache Spark applications. 

## Docker Hub

One can easily obtain the latest image using:
```
docker pull enriquegrodrigo/docker-sparkdev:latest
```

## Building the image 

For building the image:

```
git clone https://github.com/enriquegrodrigo/docker-pydata.git
docker build -t="Name of the image"
```

## Usage

To run the Spark shell: 

	docker run --rm -it -v $(pwd)/project:/home/work/project -v $(pwd)/.ivy:/sbtlib enriquegrodrigo/sparkdev  

One can also run the scala shell

	docker run --rm -it -v $(pwd)/project:/home/work/project -v $(pwd)/.ivy:/sbtlib enriquegrodrigo/sparkdev scala 


Or compile the project using sbt: 

	docker run --rm -it -v $(pwd)/project:/home/work/project -v $(pwd)/.ivy:/sbtlib enriquegrodrigo/sparkdev sbt 

One can also execut Spark applications using spark-submit: 

	docker run --rm -it -v $(pwd)/project:/home/work/project -v $(pwd)/.ivy:/sbtlib enriquegrodrigo/sparkdev spark-submit app.jar
