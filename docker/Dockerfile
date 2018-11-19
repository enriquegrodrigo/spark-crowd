FROM enriquegrodrigo/docker-sparkdev

#VOLUME /home/work/project

RUN cd /opt/ && \
	wget https://oss.sonatype.org/service/local/repositories/releases/content/com/enriquegrodrigo/spark-crowd_2.11/0.2.1/spark-crowd_2.11-0.2.1.jar -O spark-crowd-0.2.1.jar


ENTRYPOINT ["spark-shell", "--jars", "/opt/spark-crowd-0.2.1.jar", "--master", "local[*]", "-i"]
	
