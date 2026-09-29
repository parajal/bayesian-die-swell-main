TFEMPATH = /home/20220558/tfem

#
# Mdefs.mk
#
# Edit for your own system:
#
#  1) Adapt LIBBLAS to match your optimized Blas/Lapack etc.
#  2) Uncomment a line defining FC matching your compiler.
#  3) Adapt for your compiler FOPT and optionally FCHECK and FMISC.
#  4) (Optionally) Adapt METISPATH to represent your METIS installation.
#  5) (Optionally) Adapt OUTCLEAN for cleaning your output files.
#  6) (Optionally) Adapt UMFPACKPATH to represent your UMFPACK installation.
#  7) (Optionally) Adapt FPPFLAGS to set conditional compilation options.
#  8) (Optionally) Adapt FEASTROOT to represent your FEAST installation,
#
# See the file install/README.MKL in the tfem tree for more info on
# how to link to MKL for multi-threading or with 8-byte integers.
#
# See the file install/README.prepro in the tfem tree for more info on
# setting of the preprocessor for conditional compilation options.


# Some general macros

RM = rm -f
AR = ar vr
RANLIB = ranlib
OUTCLEAN = *.fig *.vtk *.out *.ps *.plt *.pos *.png *.mso

# Some tfem macros

EXTRATAGSSRC = $(TFEMPATH)/src/core/*.[fF]90 $(TFEMPATH)/src/addons/*/*.[fF]90

MODPATHINC = $(TFEMPATH)/modfiles

# Blas/Lapack

# oneMKL
# ======

MKLPATH = $(MKLROOT)/lib/intel64
# ---- PARDISO lives in MKL; flexiblas does NOT provide the PARDISO solver, so
#      linking MKL is required to use solve_system_pardiso. Load an MKL module
#      (so MKLROOT is set) and pick the line matching your compiler/integer size.
# flexiblas (BLAS/LAPACK only, NO PARDISO):
#LIBBLAS = -L$(EBROOTFLEXIBLAS)/lib -lflexiblas -lpthread
# 4-byte integer, MKL multi-threaded -- gfortran / GNU (foss) toolchain, gives PARDISO:
LIBBLAS = -L$(MKLPATH) -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
# 4-byte integer, MKL multi-threaded -- ifx / Intel compiler:
#LIBBLAS = -L$(MKLPATH) -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread
# 8-byte integer (ilp64) -- only if real(dp) arrays exceed 16GB (also set -fdefault-integer-8):
#LIBBLAS = -L$(MKLPATH) -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

# Nagfor compiled libraries
# =========================

#LIBBLAS = -llapack -lrefblas

# Apple Mx processors
# ===================

#LIBBLAS = -framework Accelerate

# the following lines are for building a library only

LIBCREATE = libdummy.a
MODPATHWRITE = .

# the following is for building a project tree including a library

ifdef PROJECTDIR
  LIBCREATE = $(PROJECTDIR)/lib/lib$(PROJECTNAME).a
  MODPATHWRITE = $(PROJECTDIR)/modfiles
  LIBSP = -L$(PROJECTDIR)/lib -l$(PROJECTNAME)
  FMODINCP = -I$(MODPATHWRITE)
endif

FMODINC = $(FMODINCP) -I$(MODPATHINC)

# Adjust the UMFPACKPATH macro to represent your UMFPACK installation

UMFPACKPATH =
ifdef UMFPACKPATH
  LIBUMFPACK = -L$(UMFPACKPATH) -lumfpack
endif

# Adjust the METISPATH macro to represent your METIS installation

METISPATH =
ifdef METISPATH
  LIBMETIS = -L$(METISPATH) -lmetis
endif

# Adjust the FEASTROOT macro to represent your FEAST installation

FEASTROOT =
ifdef FEASTROOT
  LIBFEAST = -L$(FEASTROOT)/lib/x64 -lfeast
endif

# now choose the compiler

#FC = nagfor
#FC = ifx
FC = gfortran

ifeq ($(FC),nagfor)
#  FCHECK = -nan -gline -w=uda -C -C=undefined # -C=intovf
  FMODWRITE = -mdir $(MODPATHWRITE)
  FOPT = -O
#  FOPT = -O0
#  FDEBUG = -g90
  FMISC = -f2018
#  FMISC = -f2018 -i8 # 8-byte integers
  FPP = -fpp
  FSOLVERLIB = external_nagfor
endif

ifeq ($(FC),ifx)
#  FCHECK = -check all,noarg_temp_created -traceback -warn all,nounused
  FMODWRITE = -module $(MODPATHWRITE)
  FOPT = -O3 -march=core-avx2
#  FDEBUG = -g
  FMISC = -fpe0 -stand f18
#  FMISC = -fpe0 -i8  # 8-byte integers
  FSOLVERLIB = external_ifx
endif

ifeq ($(FC),gfortran)
#  FCHECK = -Wall -fcheck=all
  FMODWRITE = -J$(MODPATHWRITE)
  FOPT = -O3 -march=native -funroll-loops -fstack-arrays
  FMISC = -std=f2018 -fall-intrinsics -ffpe-trap=zero,overflow,invalid
  FSOLVERLIB = external_gfortran
#  FMISC = #-fdefault-integer-8 # to set default 64bit integers
#  FMISC = #-finline-limit=n # to increase the inline limit.
endif

FFLAGS = $(FCHECK) $(FOPT) $(FDEBUG) $(FMISC)
LIBS = $(LIBSP) -L$(TFEMPATH)/lib -ltfem -l$(FSOLVERLIB) $(LIBUMFPACK)\
 $(LIBBLAS) $(LIBMETIS) $(LIBFEAST)

# Set preprocessor flags (see file install/README.prepro)

FPPFLAGS=$(FPP)# -DNO_LIBHSL
