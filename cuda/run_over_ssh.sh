#!/bin/bash

# this script copies the specified directory to the ssh target
# it then runs the Makefile and returns any created files to the host
# then removes all traces of compute

if [ $# -eq 0 ]
then
	echo "USAGE: $0 <dir to copy> [ssh user]"
	exit 1
fi

LOCAL_DIR=$1

SSH_USER=svatasoi
SSH_ADDR=ssh.cs.brown.edu

if [ $# -gt 1 ]
then
	SSH_USER="$2"
fi

if [ ! -f $LOCAL_DIR/Makefile ]
then
	echo "$LOCAL_DIR must contain a Makefile"
	exit 1
fi

printf "\n========Sending Files Over (>$SSH_ADDR)========\n\n"

TMP_DIR=$(mktemp -d)
rm -rf $TMP_DIR
TMP_DIR=/home/$SSH_USER/`basename $TMP_DIR`
sftp $SSH_USER@$SSH_ADDR << EOF
	mkdir $TMP_DIR
	cd $TMP_DIR
	mkdir $(basename $LOCAL_DIR)
	put -r $LOCAL_DIR
	exit
EOF

printf "\n========Running Code Remotely========\n\n"

ssh -t -t $SSH_USER@$SSH_ADDR << EOF
	cd $TMP_DIR/$LOCAL_DIR
	make >build.log 2>build.err
	make run >run.log 2>run.err
	exit
EOF

printf "\n========Retrieving Results========\n\n"

# retrieve log file and destroy tmp dir
pushd $LOCAL_DIR >/dev/null
sftp $SSH_USER@$SSH_ADDR << END
	cd $TMP_DIR/$LOCAL_DIR
	get *.log
	get *.err
	cd /home/$SSH_USER
	rm $TMP_DIR/$LOCAL_DIR/*
	rmdir $TMP_DIR/$LOCAL_DIR
	rmdir $TMP_DIR
	exit
END
popd >/dev/null
