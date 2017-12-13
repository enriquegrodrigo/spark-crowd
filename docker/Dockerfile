FROM enriquegrodrigo/docker-sparkdev

#VOLUME /home/work/project

RUN cd /opt/ && \
	wget https://oss.sonatype.org/service/local/repositories/releases/content/com/enriquegrodrigo/spark-crowd_2.11/0.1.5/spark-crowd_2.11-0.1.5.jar -O spark-crowd-0.1.5.jar 


ENTRYPOINT ["spark-shell", "--jars", "/opt/spark-crowd-0.1.5.jar", "--master", "local[*]", "-i"]
	
