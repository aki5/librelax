
CFLAGS=-O2 -fomit-frame-pointer -W -Wall

LIBRELAX_OFILES=\
	relax_aat.o\
	relax_ab.o\
	relax_ata.o\
	relax_atb.o\
	relax_gauss.o\
	relax_pinvb.o\
	relax_pinvtb.o\
	relax_solve.o\
	relax_sor.o\
	relax_svd.o\

test: relax_test
	./relax_test

librelax.a: $(LIBRELAX_OFILES)
	$(AR) r $@ $(LIBRELAX_OFILES)

relax_test: relax_test.o librelax.a
	$(CC) -o $@ relax_test.o librelax.a -lm

clean:
	rm -f relax_test *.o *.a
