#!/bin/bash -e

ZLIB=zlib-1.2.11
SCRIPT_DIR=$(dirname "$0")
HUDSON_ROOT=$(realpath ${SCRIPT_DIR}/../..)

source ${SDSOC_ROOT}/settings64.sh

cd ${SCRIPT_DIR}

echo "Downloading zlib..."
mkdir -p lib
cd lib
[ -f ${ZLIB}.tar.gz ] || wget https://www.zlib.net/${ZLIB}.tar.gz

for PLATFORM in $(uname -p) zcu102 pynq ultra96
do
  if [ ${PLATFORM} == $(uname -p) ]
  then
    export CC="gcc"
  else
    PLATFORM_DIR=$(${HUDSON_ROOT}/scripts/get_platform.py ${PLATFORM})
    export CC="sdscc -sds-pf ${PLATFORM_DIR}"
  fi

  mkdir -p ${PLATFORM}
  cd ${PLATFORM}

  echo "Copying minizip for ${PLATFORM}..."
  [ -d minizip ] || cp -r ../../cpp/minizip .

  echo "Unpacking zlib for ${PLATFORM}..."
  [ ! -d zlib ] && tar -xzf ../${ZLIB}.tar.gz && mv ${ZLIB} zlib

  echo "Building zlib for ${PLATFORM}..."
  cd zlib
  [ ! -f libz.a ] && ./configure && make
  cd ..

  echo "Building minizip for ${PLATFORM}..."
  cd minizip
  [ -f libminizip.a -a -f libaes.a ] || make MFLAGS="CC='${CC}'"
  cd ..

  cd ..
done
