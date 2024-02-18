import glob
import os, sys

MAIN_IMAGE_NAME=sys.argv[1]
TARGET_TAG="latest" if len(sys.argv) < 3 else sys.argv[2]

args=["docker manifest create {}:{}".format(MAIN_IMAGE_NAME, TARGET_TAG)]
for i in glob.glob("/tmp/images/*/*.txt"):
    with open(i, "r") as file:
        args += " --amend {}@{}".format(MAIN_IMAGE_NAME, file.readline().strip())
cmd_create="".join(args)
cmd_push="docker manifest push {}:{}".format(MAIN_IMAGE_NAME, TARGET_TAG)
os.system(cmd_create)
os.system(cmd_push)