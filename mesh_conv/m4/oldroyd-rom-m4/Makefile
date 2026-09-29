include Mdefs.mk

.SUFFIXES:

.SUFFIXES: .o .f90 .F90 .f

.f90.o:
	$(FC) -c $(FFLAGS) $(FMODINC) $<

SRC = *.f90

OBJS = update_mesh_nodes_bc.o

MAINP = extrudate_swell

all: objects programs tags

objects: $(OBJS)

programs: $(MAINP)

tags: $(SRC)
	ctags --fortran-kinds=+i $(SRC) $(EXTRATAGSSRC)

extrudate_swell.o: update_mesh_nodes_bc.o
extrudate_swell: $(OBJS) extrudate_swell.o
	$(FC) $(FFLAGS) -o $@ extrudate_swell.o $(OBJS) $(LIBS)

clean:
	$(RM)  extrudate_swell *.o *.mod *.bin tags
