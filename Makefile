
CFLAGS=-O2 -fomit-frame-pointer -W -Wall
#CFLAGS=-g

LIBRELAX_OFILES=\
	relax_aat.o\
	relax_ab.o\
	relax_ata.o\
	relax_atb.o\
	relax_conjgrad.o\
	relax_coordesc.o\
	relax_dot.o\
	relax_gauss.o\
	relax_graddesc.o\
	relax_kacz.o\
	relax_lsqr.o\
	relax_maxres.o\
	relax_pinvb.o\
	relax_pinvtb.o\
	relax_solve.o\
	relax_svd.o\

test: relax_test
	./relax_test

librelax.a: $(LIBRELAX_OFILES)
	$(AR) r $@ $(LIBRELAX_OFILES)

relax_test: relax_test.o librelax.a
	$(CC) $(CFLAGS) -o $@ relax_test.o librelax.a -lm

clean:
	rm -f relax_test *.o *.a
