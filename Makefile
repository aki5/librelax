
CFLAGS=-O2 -fomit-frame-pointer

LIBRELAX_OFILES=\
	relax.o\

test: relax_test
	./relax_test
	./relax_test jacobi

librelax.a: $(LIBRELAX_OFILES)
	$(AR) r $@ $(LIBRELAX_OFILES)

relax_test: relax_test.o librelax.a
	$(CC) -o $@ relax_test.o librelax.a

clean:
	rm -f relax_test *.o *.a
