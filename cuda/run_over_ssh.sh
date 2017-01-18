#!/bin/bash

# this script copies the specified directory to the ssh target
# it then runs the Makefile and returns any created files to the host
# then removes all traces of compute

if [ $# -eq 0 ]
then
	echo "USAGE: $0 <dir to copy> [pause] [ssh user]"
	exit 1
fi

LOCAL_DIR=$1

SSH_USER=svatasoi
SSH_ADDR=ssh.cs.brown.edu

PAUSE=0
if [ $# -gt 1 ]
then
	PAUSE="$2"
fi

if [ $# -gt 2 ]
then
	SSH_USER="$3"
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

if [ $PAUSE -ne 0 ]
then
ssh -X $SSH_USER@$SSH_ADDR
fi

printf "\n========Retrieving Results========\n\n"

# retrieve log file and destroy tmp dir
pushd $LOCAL_DIR >/dev/null
sftp $SSH_USER@$SSH_ADDR << END
	cd $TMP_DIR/$LOCAL_DIR
	get *.log
	get *.err
	exit
END
popd >/dev/null

printf "\n========Removing Remote Files========\n\n"

ssh -t -t $SSH_USER@$SSH_ADDR << EOF
	rm -rf $TMP_DIR
	exit
EOF

printf  "\n========Done!========\n"