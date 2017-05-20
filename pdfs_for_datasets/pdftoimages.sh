#!/bin/bash
for i in b1 b2 b3 b4 b5; do
	pdftoppm -rx 308 -ry 282 -png $i.pdf $i
done

z=0
for i in `ls *.png`; do
	mv $i out$z.png
	z=`expr $z + 1`
done
echo $z
