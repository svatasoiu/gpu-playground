CC = gcc
CFLAGS = -Wall -Wextra -Wunused
CFLAGS += -O3 -std=gnu99 -D_GNU_SOURCE -pthread 

OBJECTS = box_muller.o stat.o
NORED = no_reduction
ANTITHETIC = antithetic
CV = control_variate
MAIN = main.c
BASELINE = baseline
EXECS = price_option baseline_price

.PHONY: all
all: $(EXECS)

price_option: $(OBJECTS) $(NORED).c $(ANTITHETIC).c $(CV).c $(MAIN)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(BASELINE)_price: $(OBJECTS) $(NORED).c $(BASELINE)_main.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

box_muller.o: box_muller.h
stat.o: stat.h

$(OBJECTS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf *.o $(EXECS)

