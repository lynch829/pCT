TARGET_DIR := target

lein:
	lein compile && lein uberjar
	cp ${TARGET_DIR}/utils-0.1.0-SNAPSHOT-standalone.jar ${TARGET_DIR}/utils.jar
	chmod +x ${TARGET_DIR}/utils.jar
clean:
	rm -rf ${TARGET_DIR}

