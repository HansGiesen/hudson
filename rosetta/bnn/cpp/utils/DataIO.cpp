#include "DataIO.h"

Cifar10TestInputs::Cifar10TestInputs(const std::string & filename, unsigned n)
  : m_size(n*CHANNELS*ROWS*COLS)
{
  data = new float[m_size];

  DB_PRINT(2, "Opening data archive %s\n", filename.c_str());
  unzFile ar = open_unzip(filename.c_str());
  unsigned nfiles = get_nfiles_in_unzip(ar);
  assert(nfiles == 1);

  // We read m_size*4 bytes from the archive
  unsigned fsize = get_current_file_size(ar);
  assert(m_size*4 <= fsize);

  DB_PRINT(2, "Reading %u bytes\n", m_size*4);
  read_current_file(ar, (void*)data, m_size*4);
  
  unzClose(ar);
}

Cifar10TestLabels::Cifar10TestLabels(const std::string & filename, unsigned n)
  : m_size(n)
{
  data = new float[m_size];

  DB_PRINT(2, "Opening data archive %s\n", filename.c_str());
  unzFile ar = open_unzip(filename.c_str());
  unsigned nfiles = get_nfiles_in_unzip(ar);
  assert(nfiles == 1);

  // We read n*4 bytes from the archive
  unsigned fsize = get_current_file_size(ar);
  assert(m_size*4 <= fsize);

  DB_PRINT(2, "Reading %u bytes\n", m_size*4);
  read_current_file(ar, (void*)data, m_size*4);
  unzClose(ar);
}

